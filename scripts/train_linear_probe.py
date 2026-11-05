#!/usr/bin/env python
"""Train a model/layer-specific last-token safety probe from labeled JSONL.

Each JSONL row must contain ``text`` and binary ``label`` (0=safe, 1=unsafe).
The script freezes Qwen, extracts one decoder layer's final non-padding token,
trains a linear classifier, reports held-out metrics, and writes a versioned
artifact that the compiled inference wrapper validates before loading.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from torch_compile_guardrails.scoring import ProbeMetadata, save_probe


def _load_rows(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        payload = json.loads(line)
        text = payload.get("text")
        label = payload.get("label")
        if not isinstance(text, str) or label not in (0, 1):
            raise ValueError(f"{path}:{line_number}: expected text string and label 0 or 1")
        rows.append((text, int(label)))
    if len(rows) < 20 or {label for _, label in rows} != {0, 1}:
        raise ValueError("training requires at least 20 rows containing both labels")
    return rows


def _auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.numel() == 0 or negatives.numel() == 0:
        return float("nan")
    comparisons = positives[:, None] - negatives[None, :]
    return float(((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean())


@torch.inference_mode()
def _extract(
    model: Qwen3ForCausalLM,
    tokenizer,
    rows: list[tuple[str, int]],
    *,
    layer: int,
    batch_size: int,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    labels: list[int] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = tokenizer(
            [text for text, _ in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to("cuda")
        output = model(**encoded, use_cache=False, output_hidden_states=True)
        hidden = output.hidden_states[layer + 1]
        final_positions = encoded.attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        features.append(hidden[batch_indices, final_positions].float().cpu())
        labels.extend(label for _, label in batch)
    return torch.cat(features), torch.tensor(labels, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics-out", type=Path)
    args = parser.parse_args()
    if (
        not 0.0 < args.validation_fraction < 0.5
        or not 0.0 < args.test_fraction < 0.5
        or args.validation_fraction + args.test_fraction >= 0.8
    ):
        raise SystemExit("validation/test fractions must be positive and leave at least 20% train")

    rows = _load_rows(args.data)
    random.Random(args.seed).shuffle(rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, local_files_only=True
        )
        .eval()
        .to("cuda")
    )
    if not isinstance(model, Qwen3ForCausalLM):
        raise TypeError(f"expected Qwen3ForCausalLM, got {type(model).__name__}")
    layer = model.config.num_hidden_layers // 2 if args.layer is None else args.layer
    if not 0 <= layer < model.config.num_hidden_layers:
        raise SystemExit(f"layer must be in [0, {model.config.num_hidden_layers})")

    features, labels = _extract(
        model,
        tokenizer,
        rows,
        layer=layer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    split_generator = torch.Generator().manual_seed(args.seed)
    split_indices: dict[str, list[torch.Tensor]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for label in (0, 1):
        class_indices = torch.where(labels == label)[0]
        class_indices = class_indices[torch.randperm(len(class_indices), generator=split_generator)]
        validation_size = round(len(class_indices) * args.validation_fraction)
        test_size = round(len(class_indices) * args.test_fraction)
        split_indices["validation"].append(class_indices[:validation_size])
        split_indices["test"].append(class_indices[validation_size : validation_size + test_size])
        split_indices["train"].append(class_indices[validation_size + test_size :])

    def get_split(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.cat(split_indices[name])
        indices = indices[torch.randperm(len(indices), generator=split_generator)]
        return features[indices].cuda(), labels[indices].cuda()

    train_x, train_y = get_split("train")
    validation_x, validation_y = get_split("validation")
    test_x, test_y = get_split("test")

    torch.manual_seed(args.seed)
    classifier = torch.nn.Linear(model.config.hidden_size, 1, device="cuda", dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    positive_weight = (train_y == 0).sum() / (train_y == 1).sum()
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    for _ in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(classifier(train_x).squeeze(1), train_y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation_loss = float(
                loss_function(classifier(validation_x).squeeze(1), validation_y).item()
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu() for name, value in classifier.state_dict().items()
            }
    if best_state is None:
        raise AssertionError("training produced no checkpoint")
    classifier.load_state_dict(best_state)
    with torch.no_grad():
        test_logits = classifier(test_x).squeeze(1)
        test_loss = float(loss_function(test_logits, test_y).item())
        test_scores = test_logits.sigmoid().cpu()
    test_labels = test_y.cpu()
    metrics = {
        "model": args.model,
        "layer": layer,
        "dataset": args.dataset_name,
        "train_examples": len(train_x),
        "validation_examples": len(validation_x),
        "test_examples": len(test_x),
        "best_validation_bce": best_loss,
        "test_bce": test_loss,
        "test_accuracy_at_0.5": float(((test_scores >= 0.5) == test_labels.bool()).float().mean()),
        "test_auroc": _auroc(test_scores, test_labels),
        "seed": args.seed,
    }
    save_probe(
        args.output,
        best_state["weight"].squeeze(0),
        best_state["bias"].squeeze(0),
        ProbeMetadata(
            model_id=args.model,
            score_layer=layer,
            hidden_size=model.config.hidden_size,
            dataset=args.dataset_name,
        ),
    )
    print(json.dumps(metrics, indent=2))
    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
