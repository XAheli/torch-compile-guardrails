#!/usr/bin/env python
"""Run release-correct, one-token-buffered generation with Qwen3."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from torch_compile_guardrails.models import GuardedQwen3ForCausalLM
from torch_compile_guardrails.scoring import ScoringBackend
from torch_compile_guardrails.serve import generate_strict, generate_with_commit_buffer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="The central rule for safe streaming is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=1.1)
    parser.add_argument("--backend", choices=[b.value for b in ScoringBackend], default="native")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cache", choices=["dynamic", "static"], default="static")
    parser.add_argument("--policy", choices=["strict", "commit_buffer"], default="commit_buffer")
    parser.add_argument("--probe", type=Path, help="trained probe artifact")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, local_files_only=True
        )
        .eval()
        .to("cuda")
    )
    if not isinstance(base, Qwen3ForCausalLM):
        raise TypeError(f"expected Qwen3ForCausalLM, got {type(base).__name__}")
    torch.manual_seed(0)
    guarded = GuardedQwen3ForCausalLM(
        base,
        score_layer=base.config.num_hidden_layers // 2,
        scoring_backend=args.backend,
        probe_path=args.probe,
    ).eval()
    cached_forward = None
    if args.compile:
        cached_forward = torch.compile(guarded.forward_with_cache, fullgraph=True)

    prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to("cuda")
    initial_cache = None
    if args.cache == "static":
        initial_cache = StaticCache(
            config=base.config,
            max_cache_len=prompt_ids.shape[1] + args.max_new_tokens,
        )
    generate = generate_strict if args.policy == "strict" else generate_with_commit_buffer
    result = generate(
        guarded,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        threshold=args.threshold,
        eos_token_id=tokenizer.eos_token_id,
        cached_forward=cached_forward,
        initial_cache=initial_cache,
    )
    print(
        f"mode: {'compiled' if args.compile else 'eager'} / {args.cache} cache / "
        f"{result.policy.value} policy"
    )
    print(f"prompt: {args.prompt!r}")
    print(f"released: {tokenizer.decode(result.released_token_ids)!r}")
    print(f"released token ids: {result.released_token_ids}")
    print("verified:")
    for verified in result.verified_tokens:
        print(
            f"  step={verified.decode_step} token={verified.token_id} "
            f"position={verified.absolute_position} "
            f"covered_through={verified.score_covers_through_position} "
            f"text={tokenizer.decode([verified.token_id])!r} score={verified.score:.6f}"
        )
    if result.blocked_token_id is not None:
        print(
            f"blocked before release: token={result.blocked_token_id} "
            f"score={result.blocked_score:.6f}"
        )


if __name__ == "__main__":
    main()
