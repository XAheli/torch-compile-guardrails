#!/usr/bin/env python
"""Compare identical activation scoring math across compiler boundaries."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from torch_compile_guardrails.scoring import ScoringBackend, activation_score


def _time_cuda(fn, inputs, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn(*inputs)
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iterations):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn(*inputs)
        end.record()
        end.synchronize()
        samples.append(float(begin.elapsed_time(end)) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.manual_seed(0)
    device = torch.device("cuda")
    inputs = (
        torch.randn(
            args.batch,
            args.sequence,
            args.hidden,
            device=device,
            dtype=torch.bfloat16,
        ),
        torch.randn(args.hidden, device=device, dtype=torch.bfloat16),
        torch.randn((), device=device, dtype=torch.bfloat16),
    )

    rows: list[dict[str, float | str]] = []
    reference = activation_score(*inputs, ScoringBackend.NATIVE)
    for backend in ScoringBackend:

        def eager(activations, weight, bias, _backend=backend):
            return activation_score(activations, weight, bias, _backend)

        start = time.perf_counter()
        compiled = torch.compile(eager, fullgraph=True)
        compiled(*inputs)
        torch.cuda.synchronize()
        compile_seconds = time.perf_counter() - start

        eager_us = _time_cuda(eager, inputs, warmup=args.warmup, iterations=args.iterations)
        compiled_us = _time_cuda(compiled, inputs, warmup=args.warmup, iterations=args.iterations)
        actual = compiled(*inputs)
        max_error = float((actual - reference).abs().max().item())
        rows.append(
            {
                "backend": backend.value,
                "compile_seconds": compile_seconds,
                "eager_us": eager_us,
                "compiled_us": compiled_us,
                "max_abs_error": max_error,
            }
        )

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"shape: [{args.batch}, {args.sequence}, {args.hidden}] bf16")
    print(f"{'backend':<12} {'compile_s':>10} {'eager_us':>12} {'compiled_us':>12} {'max_err':>12}")
    for row in rows:
        print(
            f"{row['backend']:<12} {row['compile_seconds']:>10.3f} "
            f"{row['eager_us']:>12.2f} {row['compiled_us']:>12.2f} "
            f"{row['max_abs_error']:>12.3g}"
        )

    if args.json_out:
        payload = {
            "device": torch.cuda.get_device_name(0),
            "shape": [args.batch, args.sequence, args.hidden],
            "dtype": "bfloat16",
            "results": rows,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
