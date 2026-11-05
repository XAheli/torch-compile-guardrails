"""Opaque custom-op implementation of the activation scorer."""

import torch


@torch.library.custom_op("guardrails::opaque_activation_score", mutates_args=(), device_types=None)
def custom_op_score(
    activations: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    final_activation = activations[:, -1, :].float()
    return torch.sigmoid(final_activation @ weight.float() + bias.float())


@custom_op_score.register_fake
def _custom_op_score_fake(
    activations: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    del weight, bias
    return activations.new_empty((activations.shape[0],), dtype=torch.float32)
