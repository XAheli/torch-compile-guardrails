"""Compiler-visible scorer expressed only with native PyTorch operations."""

import torch


def native_score(
    activations: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    final_activation = activations[:, -1, :].float()
    logits = final_activation @ weight.float() + bias.float()
    return torch.sigmoid(logits)
