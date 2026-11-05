"""Release-correct greedy generation policies for activation scores."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import torch
from transformers.cache_utils import Cache

from torch_compile_guardrails.models import GuardedQwen3ForCausalLM


class EnforcementPolicy(StrEnum):
    """When certified tokens cross the serving boundary."""

    STRICT = "strict"
    COMMIT_BUFFER = "commit_buffer"


@dataclass(frozen=True)
class VerifiedToken:
    token_id: int
    score: float
    decode_step: int
    absolute_position: int
    score_covers_through_position: int


@dataclass
class CommitBufferResult:
    policy: EnforcementPolicy = EnforcementPolicy.COMMIT_BUFFER
    released_token_ids: list[int] = field(default_factory=list)
    verified_tokens: list[VerifiedToken] = field(default_factory=list)
    blocked_token_id: int | None = None
    blocked_score: float | None = None


@torch.no_grad()
def generate_with_commit_buffer(
    model: GuardedQwen3ForCausalLM,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    threshold: float,
    eos_token_id: int | None = None,
    cached_forward: Callable[
        [torch.Tensor, torch.Tensor | None, Cache | None],
        tuple[torch.Tensor, torch.Tensor, Cache],
    ]
    | None = None,
    initial_cache: Cache | None = None,
) -> CommitBufferResult:
    """Release each token after a score covering that token passes policy."""
    return _generate_verified(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        threshold=threshold,
        eos_token_id=eos_token_id,
        cached_forward=cached_forward,
        initial_cache=initial_cache,
        policy=EnforcementPolicy.COMMIT_BUFFER,
    )


@torch.no_grad()
def generate_strict(
    model: GuardedQwen3ForCausalLM,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    threshold: float,
    eos_token_id: int | None = None,
    cached_forward: Callable[
        [torch.Tensor, torch.Tensor | None, Cache | None],
        tuple[torch.Tensor, torch.Tensor, Cache],
    ]
    | None = None,
    initial_cache: Cache | None = None,
) -> CommitBufferResult:
    """Quarantine the complete response until every generated token passes."""
    return _generate_verified(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        threshold=threshold,
        eos_token_id=eos_token_id,
        cached_forward=cached_forward,
        initial_cache=initial_cache,
        policy=EnforcementPolicy.STRICT,
    )


def _generate_verified(
    model: GuardedQwen3ForCausalLM,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    threshold: float,
    eos_token_id: int | None,
    cached_forward: Callable[
        [torch.Tensor, torch.Tensor | None, Cache | None],
        tuple[torch.Tensor, torch.Tensor, Cache],
    ]
    | None,
    initial_cache: Cache | None,
    policy: EnforcementPolicy,
) -> CommitBufferResult:
    """Shared decode engine with an explicit token-to-score coverage ledger."""
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("commit-buffer reference currently requires batch size one")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    result = CommitBufferResult(policy=policy)
    if max_new_tokens == 0:
        return result

    # StaticCache must be allocated and have its addresses marked before the
    # compiled region. Prefill therefore stays eager; only the fixed-shape,
    # one-token decode step uses the optional compiled callable.
    logits, _, cache = model.forward_with_cache(prompt_ids, None, initial_cache)
    decode_forward = cached_forward or model.forward_with_cache
    pending = logits[:, -1, :].argmax(dim=-1)

    for decode_step in range(max_new_tokens):
        logits, score_tensor, cache = decode_forward(pending.unsqueeze(1), None, cache)
        score = float(score_tensor[0].item())
        token_id = int(pending[0].item())
        absolute_position = prompt_ids.shape[1] + decode_step
        result.verified_tokens.append(
            VerifiedToken(
                token_id=token_id,
                score=score,
                decode_step=decode_step,
                absolute_position=absolute_position,
                score_covers_through_position=absolute_position,
            )
        )
        if score >= threshold:
            result.blocked_token_id = token_id
            result.blocked_score = score
            break

        if policy is EnforcementPolicy.COMMIT_BUFFER:
            result.released_token_ids.append(token_id)
        if eos_token_id is not None and token_id == eos_token_id:
            break
        pending = logits[:, -1, :].argmax(dim=-1)

    if policy is EnforcementPolicy.STRICT and result.blocked_token_id is None:
        result.released_token_ids.extend(token.token_id for token in result.verified_tokens)
    return result
