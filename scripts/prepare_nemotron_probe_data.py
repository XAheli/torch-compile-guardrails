#!/usr/bin/env python
"""Create a deterministic balanced JSONL subset of NVIDIA Nemotron safety data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", choices=["prompt", "response"], default="response")
    args = parser.parse_args()
    if args.examples_per_class <= 0:
        raise SystemExit("examples-per-class must be positive")

    stream = load_dataset(
        "nvidia/Nemotron-3.5-Content-Safety-Dataset",
        split="train",
        streaming=True,
    )
    by_label: dict[str, list[dict[str, str | int]]] = {"safe": [], "unsafe": []}
    for row in stream:
        label = row["input_label"] if args.target == "prompt" else row["response_label"]
        if label not in by_label or row["language"] != "en" or not row["prompt"]:
            continue
        if args.target == "response":
            if not row["response"]:
                continue
            text = f"User: {row['prompt']}\nAssistant: {row['response']}"
        else:
            text = row["prompt"]
        by_label[label].append(
            {
                "text": text,
                "label": 0 if label == "safe" else 1,
                "row_id": row["row_id"],
                "target": args.target,
            }
        )
        if all(len(rows) >= args.examples_per_class for rows in by_label.values()):
            break
    if any(len(rows) < args.examples_per_class for rows in by_label.values()):
        counts = {label: len(rows) for label, rows in by_label.items()}
        raise RuntimeError(f"dataset stream ended before balanced subset was filled: {counts}")

    rows = [
        row for label_rows in by_label.values() for row in label_rows[: args.examples_per_class]
    ]
    random.Random(args.seed).shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"wrote {len(rows)} balanced examples to {args.output}")


if __name__ == "__main__":
    main()
