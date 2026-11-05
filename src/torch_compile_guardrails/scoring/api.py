"""Shared contract for activation safety scorers.

All backends implement the same operation: select the final sequence position,
apply a linear probe in fp32, and return one sigmoid score per batch element.
Keeping the math identical makes compiler and kernel comparisons meaningful.
"""

from __future__ import annotations

from enum import StrEnum

import torch

from .custom_op import custom_op_score
from .native import native_score
from .triton_op import triton_score


class ScoringBackend(StrEnum):
    NATIVE = "native"
    CUSTOM_OP = "custom_op"
    TRITON_OP = "triton_op"


def validate_inputs(activations: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> None:
    if activations.ndim != 3:
        raise ValueError("activations must have shape [batch, sequence, hidden]")
    if weight.ndim != 1:
        raise ValueError("weight must have shape [hidden]")
    if bias.ndim != 0:
        raise ValueError("bias must be a scalar tensor")
    if activations.shape[-1] != weight.shape[0]:
        raise ValueError("activation hidden size must match probe weight")
    if activations.shape[1] == 0:
        raise ValueError("sequence dimension must be non-empty")
    if activations.device != weight.device or activations.device != bias.device:
        raise ValueError("activations, weight, and bias must share a device")


def activation_score(
    activations: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    backend: ScoringBackend | str = ScoringBackend.NATIVE,
) -> torch.Tensor:
    """Return one fp32 safety score per batch element."""
    validate_inputs(activations, weight, bias)
    backend = ScoringBackend(backend)
    if backend is ScoringBackend.NATIVE:
        return native_score(activations, weight, bias)
    if backend is ScoringBackend.CUSTOM_OP:
        return custom_op_score(activations, weight, bias)
    if backend is ScoringBackend.TRITON_OP:
        return triton_score(activations, weight, bias)
    raise AssertionError(f"unhandled backend: {backend}")
