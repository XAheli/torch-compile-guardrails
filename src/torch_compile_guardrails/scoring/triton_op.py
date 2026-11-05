"""Compiler-visible Triton implementation of the activation scorer."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _activation_score_kernel(
    activations,
    weight,
    bias,
    output,
    sequence,
    hidden,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < hidden
    activation_offset = (batch * sequence + (sequence - 1)) * hidden
    values = tl.load(activations + activation_offset + offsets, mask=mask, other=0.0)
    weights = tl.load(weight + offsets, mask=mask, other=0.0)
    logit = tl.sum(values.to(tl.float32) * weights.to(tl.float32), axis=0)
    logit += tl.load(bias).to(tl.float32)
    score = 1.0 / (1.0 + tl.exp(-logit))
    tl.store(output + batch, score)


@torch.library.triton_op("guardrails::triton_activation_score", mutates_args={})
def _triton_score_op(
    activations: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    activations = activations.contiguous()
    weight = weight.contiguous()
    output = torch.empty((activations.shape[0],), device=activations.device, dtype=torch.float32)
    block = triton.next_power_of_2(activations.shape[-1])
    torch.library.wrap_triton(_activation_score_kernel)[(activations.shape[0],)](
        activations,
        weight,
        bias,
        output,
        activations.shape[1],
        activations.shape[2],
        BLOCK=block,
    )
    return output


@_triton_score_op.register_kernel("cpu")
def _triton_score_cpu(
    activations: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    final_activation = activations[:, -1, :].float()
    return torch.sigmoid(final_activation @ weight.float() + bias.float())


def triton_score(
    activations: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return _triton_score_op(activations, weight, bias)
