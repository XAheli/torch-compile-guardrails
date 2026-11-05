#!/usr/bin/env python
"""Test manual CUDA Graph replay for fixed-shape Qwen cached decode.

This experiment deliberately uses a fixed token tensor and StaticCache. It
measures a kernel-launch regime, not an autoregressive sampler: changing token
IDs on the host or enforcing a score requires additional synchronization and
is outside the captured graph.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from torch_compile_guardrails.models import GuardedQwen3ForCausalLM


def _time_replays(replay, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        replay()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="CUDA Graph launch experiment:")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.iterations <= 0 or args.repeats <= 0:
        raise SystemExit("batch-size, iterations, and repeats must be positive")

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
    guarded = GuardedQwen3ForCausalLM(base, score_layer=base.config.num_hidden_layers // 2).eval()
    prompt = tokenizer(args.prompt, return_tensors="pt").input_ids.to("cuda")
    prompt = prompt.expand(args.batch_size, -1).contiguous()
    token = torch.full(
        (args.batch_size, 1), tokenizer.eos_token_id - 1, dtype=torch.long, device="cuda"
    )
    capacity = prompt.shape[1] + args.iterations + 8

    with torch.no_grad():
        eager_cache = StaticCache(config=base.config, max_cache_len=capacity)
        guarded.forward_with_cache(prompt, None, eager_cache)
        guarded.forward_with_cache(token, None, eager_cache)
        eager_cache.reset()
        guarded.forward_with_cache(prompt, None, eager_cache)

        graph_cache = StaticCache(config=base.config, max_cache_len=capacity)
        guarded.forward_with_cache(prompt, None, graph_cache)
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                guarded.forward_with_cache(token, None, graph_cache)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        graph_cache.reset()
        guarded.forward_with_cache(prompt, None, graph_cache)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_logits, graph_score, _ = guarded.forward_with_cache(token, None, graph_cache)

        eager_cache.reset()
        guarded.forward_with_cache(prompt, None, eager_cache)
        eager_logits, eager_score, _ = guarded.forward_with_cache(token, None, eager_cache)
        graph_cache.reset()
        guarded.forward_with_cache(prompt, None, graph_cache)
        graph.replay()
        torch.cuda.synchronize()
        if not torch.isfinite(graph_score).all() or not torch.isfinite(graph_logits).all():
            raise RuntimeError("CUDA Graph replay produced non-finite output")
        logits_error = float((graph_logits - eager_logits).abs().max().item())
        score_error = float((graph_score - eager_score).abs().max().item())
        torch.testing.assert_close(graph_logits, eager_logits, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(graph_score, eager_score, rtol=2e-3, atol=2e-3)

        eager_samples: list[float] = []
        graph_samples: list[float] = []
        for repeat in range(args.repeats):
            order = ("eager", "graph") if repeat % 2 == 0 else ("graph", "eager")
            for mode in order:
                if mode == "eager":
                    eager_cache.reset()
                    guarded.forward_with_cache(prompt, None, eager_cache)

                    def replay_eager() -> None:
                        guarded.forward_with_cache(token, None, eager_cache)

                    eager_samples.append(_time_replays(replay_eager, args.iterations))
                else:
                    graph_cache.reset()
                    guarded.forward_with_cache(prompt, None, graph_cache)
                    graph_samples.append(_time_replays(graph.replay, args.iterations))

    eager_median = float(statistics.median(eager_samples))
    graph_median = float(statistics.median(graph_samples))
    speedup = eager_median / graph_median
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"batch/iterations/repeats: {args.batch_size} / {args.iterations} / {args.repeats}")
    print(f"eager launch median: {eager_median:.2f} us/step")
    print(f"CUDA Graph median: {graph_median:.2f} us/step")
    print(f"replay speedup: {speedup:.3f}x")
    print(f"correctness: logits_max_error={logits_error:.3g} score_max_error={score_error:.3g}")

    if args.json_out:
        payload = {
            "device": torch.cuda.get_device_name(0),
            "model": args.model,
            "batch_size": args.batch_size,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "eager_us_per_step": eager_samples,
            "cuda_graph_us_per_step": graph_samples,
            "eager_median_us_per_step": eager_median,
            "cuda_graph_median_us_per_step": graph_median,
            "speedup": speedup,
            "logits_max_error": logits_error,
            "score_max_error": score_error,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
