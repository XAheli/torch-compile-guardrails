#!/usr/bin/env python
"""Run a real Qwen prefill that returns next-token logits and an in-graph score."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from torch_compile_guardrails.models import GuardedQwen3ForCausalLM
from torch_compile_guardrails.scoring import ScoringBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--prompt", default="Explain why release ordering matters in streaming inference."
    )
    parser.add_argument("--score-layer", type=int)
    parser.add_argument("--backend", choices=[b.value for b in ScoringBackend], default="native")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--probe", type=Path, help="trained probe artifact")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        .eval()
        .to("cuda")
    )
    if not isinstance(base, Qwen3ForCausalLM):
        raise TypeError(f"expected Qwen3ForCausalLM, got {type(base).__name__}")

    score_layer = args.score_layer
    if score_layer is None:
        score_layer = base.config.num_hidden_layers // 2
    torch.manual_seed(0)
    guarded = GuardedQwen3ForCausalLM(
        base,
        score_layer=score_layer,
        scoring_backend=args.backend,
        probe_path=args.probe,
    ).eval()
    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to("cuda")
    attention_mask = encoded.attention_mask.to("cuda")

    runner = guarded
    mode = "eager"
    compile_seconds = 0.0
    if not args.no_compile:
        runner = torch.compile(guarded, fullgraph=True)
        start = time.perf_counter()
        with torch.no_grad():
            runner(input_ids, attention_mask)
        torch.cuda.synchronize()
        compile_seconds = time.perf_counter() - start
        mode = "compiled"

    with torch.no_grad():
        for _ in range(args.warmup):
            runner(input_ids, attention_mask)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        start.record()
        logits, score = runner(input_ids, attention_mask)
        end.record()
        end.synchronize()

    next_token = logits[:, -1, :].argmax(dim=-1)
    print(f"model: {args.model}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"mode/backend: {mode}/{args.backend}")
    print(f"score layer: {score_layer}")
    print(f"input tokens: {input_ids.shape[1]}")
    print(f"compile: {compile_seconds:.3f} s")
    print(f"prefill: {start.elapsed_time(end):.3f} ms")
    print(f"next token: {tokenizer.decode(next_token)}")
    probe_kind = "trained probe" if args.probe else "synthetic probe"
    print(f"{probe_kind} score: {float(score[0].item()):.6f}")
    print("coverage: score=input prefix; logits=next token")


if __name__ == "__main__":
    main()
