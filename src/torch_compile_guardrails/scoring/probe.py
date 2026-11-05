"""Versioned linear-probe artifacts tied to one model checkpoint and layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ProbeMetadata:
    model_id: str
    score_layer: int
    hidden_size: int
    dataset: str
    pooling: str = "last_token"
    format_version: int = 1


def save_probe(
    path: str | Path,
    weight: torch.Tensor,
    bias: torch.Tensor,
    metadata: ProbeMetadata,
) -> None:
    if weight.shape != (metadata.hidden_size,) or bias.shape != ():
        raise ValueError("probe tensor shapes do not match metadata")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "weight": weight.detach().float().cpu(),
            "bias": bias.detach().float().cpu(),
            "metadata": asdict(metadata),
        },
        path,
    )


def load_probe(
    path: str | Path,
    *,
    expected_hidden_size: int,
    expected_layer: int,
    expected_model_id: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, ProbeMetadata]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not {"weight", "bias", "metadata"} <= payload.keys():
        raise ValueError("invalid probe artifact")
    metadata = ProbeMetadata(**payload["metadata"])
    if metadata.format_version != 1:
        raise ValueError(f"unsupported probe format version: {metadata.format_version}")
    if metadata.hidden_size != expected_hidden_size:
        raise ValueError(
            f"probe hidden size {metadata.hidden_size} does not match model {expected_hidden_size}"
        )
    if metadata.score_layer != expected_layer:
        raise ValueError(
            f"probe layer {metadata.score_layer} does not match configured layer {expected_layer}"
        )
    if expected_model_id is not None and metadata.model_id != expected_model_id:
        raise ValueError(f"probe model {metadata.model_id!r} does not match {expected_model_id!r}")
    weight = payload["weight"]
    bias = payload["bias"]
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise ValueError("probe weight and bias must be tensors")
    if weight.shape != (expected_hidden_size,) or bias.shape != ():
        raise ValueError("probe tensor shapes do not match metadata")
    return weight, bias, metadata
