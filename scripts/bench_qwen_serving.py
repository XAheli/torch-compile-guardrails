#!/usr/bin/env python
"""Measure real-Qwen prefill, cached decode, and score synchronization.

The decode comparison holds model, probe, token, cache capacity, and number of
steps fixed. ``pipeline`` reads no host scalar and synchronizes once after the
burst. ``per_step_item`` reads the score after every decode step. This isolates
the serving-boundary synchronization cost without claiming that the pipeline
implements an enforcement policy.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from torch_compile_guardrails.models import GuardedQwen3ForCausalLM
from torch_compile_guardrails.scoring import ScoringBackend


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="Explain safe streaming in one sentence:")
    parser.add_argument("--backend", choices=[b.value for b in ScoringBackend], default="native")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.steps <= 0 or args.repeats <= 0:
        raise SystemExit("batch-size, steps, and repeats must be positive")

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
    ).eval()
    prompt = tokenizer(args.prompt, return_tensors="pt").input_ids.to("cuda")
    prompt = prompt.expand(args.batch_size, -1).contiguous()
    token = torch.full(
        (args.batch_size, 1), tokenizer.eos_token_id - 1, dtype=torch.long, device="cuda"
    )
    capacity = prompt.shape[1] + args.steps + 2

    prefill_ms: list[float] = []
    with torch.no_grad():
        warmup_cache = StaticCache(config=base.config, max_cache_len=capacity)
        guarded.forward_with_cache(prompt, None, warmup_cache)
        torch.cuda.synchronize()
        for _ in range(args.repeats):
            cache = StaticCache(config=base.config, max_cache_len=capacity)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            guarded.forward_with_cache(prompt, None, cache)
            end.record()
            end.synchronize()
            prefill_ms.append(float(start.elapsed_time(end)))

        compile_cache = StaticCache(config=base.config, max_cache_len=capacity)
        guarded.forward_with_cache(prompt, None, compile_cache)
        decode = guarded.forward_with_cache
        compile_seconds = 0.0
        if args.compile:
            decode = torch.compile(decode, fullgraph=True)
            torch.cuda.synchronize()
            compile_start = time.perf_counter()
            decode(token, None, compile_cache)
            torch.cuda.synchronize()
            compile_seconds = time.perf_counter() - compile_start

        samples: dict[str, list[dict[str, float]]] = {"pipeline": [], "per_step_item": []}
        modes = list(samples)
        for repeat in range(args.repeats):
            for mode in modes[repeat % 2 :] + modes[: repeat % 2]:
                cache = StaticCache(config=base.config, max_cache_len=capacity)
                guarded.forward_with_cache(prompt, None, cache)
                torch.cuda.synchronize()
                gpu_start = torch.cuda.Event(enable_timing=True)
                gpu_end = torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                gpu_start.record()
                last_score = None
                for _ in range(args.steps):
                    _, score, cache = decode(token, None, cache)
                    if mode == "per_step_item":
                        last_score = float(score[0].item())
                gpu_end.record()
                gpu_end.synchronize()
                wall_ms = (time.perf_counter() - wall_start) * 1000.0
                samples[mode].append(
                    {
                        "wall_ms_per_step": wall_ms / args.steps,
                        "gpu_ms_per_step": float(gpu_start.elapsed_time(gpu_end)) / args.steps,
                        "last_score": last_score,
                    }
                )

    summary = {
        mode: {
            "median_wall_ms_per_step": _median(
                [sample["wall_ms_per_step"] for sample in mode_samples]
            ),
            "median_gpu_ms_per_step": _median(
                [sample["gpu_ms_per_step"] for sample in mode_samples]
            ),
        }
        for mode, mode_samples in samples.items()
    }
    summary["sync_delta_ms_per_step"] = (
        summary["per_step_item"]["median_wall_ms_per_step"]
        - summary["pipeline"]["median_wall_ms_per_step"]
    )

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"model/backend: {args.model} / {args.backend}")
    print(
        f"execution/batch/steps: {'compiled' if args.compile else 'eager'} / {args.batch_size} / {args.steps}"
    )
    print(f"compile: {compile_seconds:.3f} s")
    print(f"prefill median: {_median(prefill_ms):.3f} ms")
    for mode in modes:
        print(
            f"{mode}: wall={summary[mode]['median_wall_ms_per_step']:.4f} ms/step "
            f"gpu={summary[mode]['median_gpu_ms_per_step']:.4f} ms/step"
        )
    print(f"per-step host-read delta: {summary['sync_delta_ms_per_step']:.4f} ms/step")

    if args.json_out:
        payload = {
            "device": torch.cuda.get_device_name(0),
            "model": args.model,
            "backend": args.backend,
            "compiled": args.compile,
            "batch_size": args.batch_size,
            "prompt_tokens": prompt.shape[1],
            "steps": args.steps,
            "repeats": args.repeats,
            "compile_seconds": compile_seconds,
            "prefill_ms": prefill_ms,
            "samples": samples,
            "summary": summary,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
