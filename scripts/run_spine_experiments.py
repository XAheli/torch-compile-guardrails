#!/usr/bin/env python
"""Run the two spine experiments for the SciPy talk thesis.

1. One-token certification gap (deterministic policy simulation)
2. ``.item()`` synchronization tax (CUDA burst microbenchmark)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torch_compile_guardrails.experiments.certification_gap import (
    DEFAULT_SCORES,
    DEFAULT_THRESHOLD,
    compare_policies,
)
from torch_compile_guardrails.experiments.sync_tax import (
    compare_sync_modes,
    sync_tax_summary,
)


def _print_gap(results: dict) -> None:
    print("=== 1) One-token certification gap ===")
    print(f"scores={list(DEFAULT_SCORES)}  threshold={DEFAULT_THRESHOLD}")
    print(f"{'policy':<28} {'released':<18} {'blocked':<8} {'leaked':<8} correct")
    for name, res in results.items():
        released = ",".join(map(str, res.released_tokens)) or "-"
        blocked = "-" if res.blocked_token is None else str(res.blocked_token)
        leaked = ",".join(map(str, res.leaked_unsafe_tokens)) or "-"
        print(f"{name:<28} {released:<18} {blocked:<8} {leaked:<8} {res.is_correct}")
    naive = results["release_before_certify"]
    buffered = results["one_token_commit_buffer"]
    print()
    print(
        f"gap: naive leaked {naive.gap_tokens} unsafe token(s); "
        f"commit-buffer leaked {buffered.gap_tokens}."
    )


def _print_sync(results: dict, summary: dict[str, float]) -> None:
    print()
    print("=== 2) .item() synchronization tax ===")
    print(
        "Fair timing: host wall clock over a burst. "
        "pipeline = N launches + 1 sync; per_step_item = .item() each step."
    )
    print(f"{'mode':<24} {'med_ms/step':>12} {'mean_ms/step':>13} {'p90_ms/step':>12}")
    for name, res in results.items():
        print(
            f"{name:<24} {res.median_ms_per_step:12.4f} "
            f"{res.mean_ms_per_step:13.4f} {res.p90_ms_per_step:12.4f}"
        )
    print()
    print("spine quantities (median ms / step):")
    print(f"  score pipeline:     {summary['score_pipeline_ms']:.4f} ms")
    print(f"  score + .item():    {summary['score_per_step_item_ms']:.4f} ms")
    print(
        f"  sync tax (score):   {summary['score_sync_tax_ms']:.4f} ms "
        f"({summary['score_sync_tax_ratio']:.2f}x pipeline)"
    )
    print(
        f"  sync tax (decode):  {summary['decode_sync_tax_ms']:.4f} ms "
        f"({summary['decode_sync_tax_ratio']:.2f}x pipeline)"
    )
    print()
    if summary["score_sync_tax_ms"] > summary["score_pipeline_ms"]:
        print("SPINE: per-step .item() sync tax exceeds pipelined scoring itself.")
    else:
        print("SPINE UNCLEAR: sync tax did not exceed pipelined scoring in this run.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bursts", type=int, default=50)
    parser.add_argument("--steps-per-burst", type=int, default=64)
    parser.add_argument("--warmup-bursts", type=int, default=5)
    parser.add_argument("--decode-dim", type=int, default=1024)
    parser.add_argument("--score-dim", type=int, default=256)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="optional path to write machine-readable results",
    )
    parser.add_argument(
        "--skip-cuda",
        action="store_true",
        help="only run the certification-gap simulation",
    )
    args = parser.parse_args(argv)

    gap = compare_policies()
    _print_gap(gap)

    payload: dict = {
        "certification_gap": {
            name: {
                "released_tokens": res.released_tokens,
                "blocked_token": res.blocked_token,
                "leaked_unsafe_tokens": res.leaked_unsafe_tokens,
                "is_correct": res.is_correct,
                "gap_tokens": res.gap_tokens,
            }
            for name, res in gap.items()
        }
    }

    if not args.skip_cuda:
        sync = compare_sync_modes(
            bursts=args.bursts,
            steps_per_burst=args.steps_per_burst,
            warmup_bursts=args.warmup_bursts,
            decode_dim=args.decode_dim,
            score_dim=args.score_dim,
        )
        summary = sync_tax_summary(sync)
        _print_sync(sync, summary)
        payload["sync_tax"] = {name: res.as_dict() for name, res in sync.items()}
        payload["sync_tax_summary"] = summary

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nWrote {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
