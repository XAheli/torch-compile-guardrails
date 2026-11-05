#!/usr/bin/env python
"""Controlled Qwen decode benchmark for activation-scoring backends.

This is a kernel-path benchmark, not a serving benchmark: it intentionally does
not sample tokens, call ``.item()`` in the timed region, or apply a host policy.
Every measurement uses eager prefill, a fresh fixed-capacity StaticCache, the
same fixed decode token, and the same model/probe weights.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch._dynamo.utils import counters
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from torch_compile_guardrails.models import GuardedQwen3ForCausalLM
from torch_compile_guardrails.scoring import ScoringBackend


@dataclass(frozen=True)
class DecodeMeasurement:
    backend: str
    execution: str
    first_call_seconds: float
    median_us: float
    mean_us: float
    p90_us: float
    minimum_us: float
    maximum_us: float
    repeat_median_min_us: float
    repeat_median_max_us: float
    repeat_overhead_median_percent: float
    repeat_overhead_min_percent: float
    repeat_overhead_max_percent: float
    unique_graphs: int
    score: float
    max_score_error: float
    max_logits_error: float
    overhead_vs_none_percent: float


@dataclass
class PreparedRunner:
    backend: str
    execution: str
    wrapper: GuardedQwen3ForCausalLM
    forward: object
    cache: StaticCache
    first_call_seconds: float
    unique_graphs: int
    samples_us: list[float]
    repeat_medians_us: list[float]
    score: torch.Tensor | None = None
    logits: torch.Tensor | None = None


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _fresh_cache(model: Qwen3ForCausalLM, capacity: int) -> StaticCache:
    return StaticCache(config=model.config, max_cache_len=capacity)


def _time_round(
    forward,
    token: torch.Tensor,
    cache: StaticCache,
    *,
    warmup: int,
    iterations: int,
) -> tuple[list[float], torch.Tensor, torch.Tensor]:
    for _ in range(warmup):
        logits, score, cache = forward(token, None, cache)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        _, score, cache = forward(token, None, cache)
        end.record()
    ends[-1].synchronize()
    samples_us = [
        float(start.elapsed_time(end)) * 1000.0 for start, end in zip(starts, ends, strict=True)
    ]
    return samples_us, logits.detach(), score.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="Explain the release invariant briefly:")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--cache-capacity", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0 or args.batch_size <= 0:
        raise SystemExit("warmup must be non-negative; iterations and repeats must be positive")

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

    prompt = tokenizer(args.prompt, return_tensors="pt").input_ids.to("cuda")
    prompt = prompt.expand(args.batch_size, -1).contiguous()
    token = torch.full(
        (args.batch_size, 1), tokenizer.eos_token_id - 1, device="cuda", dtype=torch.long
    )
    required_capacity = prompt.shape[1] + 1 + args.warmup + args.iterations
    if args.cache_capacity < required_capacity:
        raise SystemExit(
            f"cache capacity {args.cache_capacity} is too small; need at least {required_capacity}"
        )
    capacity = args.cache_capacity
    layer = base.config.num_hidden_layers // 2

    torch.manual_seed(0)
    wrappers = {
        "none": GuardedQwen3ForCausalLM(base, score_layer=None).eval(),
        **{
            backend.value: GuardedQwen3ForCausalLM(
                base, score_layer=layer, scoring_backend=backend
            ).eval()
            for backend in ScoringBackend
        },
    }
    native_scorer = wrappers[ScoringBackend.NATIVE.value].scorer
    if native_scorer is None:
        raise AssertionError("native scorer was not created")
    reference_scorer = native_scorer.state_dict()
    for name, wrapper in wrappers.items():
        if name != "none":
            if wrapper.scorer is None:
                raise AssertionError(f"{name} scorer was not created")
            wrapper.scorer.load_state_dict(reference_scorer)

    prepared: dict[str, list[PreparedRunner]] = {"eager": [], "compiled": []}
    with torch.no_grad():
        for execution in ("eager", "compiled"):
            for backend, wrapper in wrappers.items():
                cache = _fresh_cache(base, capacity)
                wrapper.forward_with_cache(prompt, None, cache)
                forward = wrapper.forward_with_cache
                if execution == "compiled":
                    forward = torch.compile(forward, fullgraph=True)

                counters.clear()
                torch.cuda.synchronize()
                first_start = time.perf_counter()
                forward(token, None, cache)
                torch.cuda.synchronize()
                first_call = time.perf_counter() - first_start
                prepared[execution].append(
                    PreparedRunner(
                        backend=backend,
                        execution=execution,
                        wrapper=wrapper,
                        forward=forward,
                        cache=cache,
                        first_call_seconds=first_call,
                        unique_graphs=int(counters["stats"]["unique_graphs"]),
                        samples_us=[],
                        repeat_medians_us=[],
                    )
                )

        # Rotate order every round so thermal/clock drift is distributed across
        # variants instead of being confounded with a fixed backend order.
        for runners in prepared.values():
            for repeat in range(args.repeats):
                shift = repeat % len(runners)
                round_order = runners[shift:] + runners[:shift]
                for runner in round_order:
                    runner.cache.reset()
                    runner.wrapper.forward_with_cache(prompt, None, runner.cache)
                    samples, logits, score = _time_round(
                        runner.forward,
                        token,
                        runner.cache,
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                    runner.samples_us.extend(samples)
                    runner.repeat_medians_us.append(statistics.median(samples))
                    runner.logits = logits
                    runner.score = score

    measurements: list[DecodeMeasurement] = []
    for execution, runners in prepared.items():
        baseline = next(runner for runner in runners if runner.backend == "none")
        native = next(runner for runner in runners if runner.backend == ScoringBackend.NATIVE.value)
        if native.score is None:
            raise AssertionError("native runner produced no score")
        baseline_median = statistics.median(baseline.samples_us)
        for runner in runners:
            if runner.score is None or runner.logits is None or baseline.logits is None:
                raise AssertionError(f"{runner.backend} runner produced incomplete outputs")
            median_us = statistics.median(runner.samples_us)
            score_error = 0.0
            if runner.backend != "none":
                score_error = float((runner.score - native.score).abs().max().item())
            logits_error = float((runner.logits - baseline.logits).abs().max().item())
            repeat_overheads = [
                (sample / baseline_sample - 1.0) * 100.0
                for sample, baseline_sample in zip(
                    runner.repeat_medians_us,
                    baseline.repeat_medians_us,
                    strict=True,
                )
            ]
            overhead_percent = (
                (median_us / baseline_median - 1.0) * 100.0 if runner.backend != "none" else 0.0
            )
            measurements.append(
                DecodeMeasurement(
                    backend=runner.backend,
                    execution=execution,
                    first_call_seconds=runner.first_call_seconds,
                    median_us=median_us,
                    mean_us=statistics.fmean(runner.samples_us),
                    p90_us=_percentile(runner.samples_us, 0.9),
                    minimum_us=min(runner.samples_us),
                    maximum_us=max(runner.samples_us),
                    repeat_median_min_us=min(runner.repeat_medians_us),
                    repeat_median_max_us=max(runner.repeat_medians_us),
                    repeat_overhead_median_percent=statistics.median(repeat_overheads),
                    repeat_overhead_min_percent=min(repeat_overheads),
                    repeat_overhead_max_percent=max(repeat_overheads),
                    unique_graphs=runner.unique_graphs,
                    score=float(runner.score[0].item()),
                    max_score_error=score_error,
                    max_logits_error=logits_error,
                    overhead_vs_none_percent=overhead_percent,
                )
            )

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"model/layer: {args.model} / {layer}")
    print(f"batch/prompt/cache: {args.batch_size} / {prompt.shape[1]} tokens / capacity {capacity}")
    print(f"warmup/timed/repeats: {args.warmup} / {args.iterations} / {args.repeats}")
    print(
        f"{'execution':<10} {'backend':<10} {'first_s':>9} {'median_us':>11} "
        f"{'p90_us':>10} {'round_range_us':>23} {'paired_ovh':>11} {'graphs':>7} "
        f"{'score_err':>11} {'logits_err':>11}"
    )
    for row in measurements:
        print(
            f"{row.execution:<10} {row.backend:<10} {row.first_call_seconds:9.3f} "
            f"{row.median_us:11.2f} {row.p90_us:10.2f} "
            f"{row.repeat_median_min_us:10.2f}-{row.repeat_median_max_us:<10.2f} "
            f"{row.repeat_overhead_median_percent:10.2f}% {row.unique_graphs:7d} "
            f"{row.max_score_error:11.3g} {row.max_logits_error:11.3g}"
        )

    if args.json_out:
        payload = {
            "device": torch.cuda.get_device_name(0),
            "model": args.model,
            "score_layer": layer,
            "prompt_tokens": prompt.shape[1],
            "batch_size": args.batch_size,
            "cache_capacity": capacity,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "repeat_medians_us": {
                execution: {runner.backend: runner.repeat_medians_us for runner in runners}
                for execution, runners in prepared.items()
            },
            "measurements": [asdict(row) for row in measurements],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
