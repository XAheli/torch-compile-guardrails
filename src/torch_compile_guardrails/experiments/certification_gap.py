"""Deterministic one-token certification-gap experiment.

Streaming decode produces a token and a GPU-side safety score each step.
Release correctness requires the host to know the score *before* the token
leaves the serving boundary.

Three policies:

* ``release_before_certify`` — emit immediately, certify later (naive async).
  Leaks the first unsafe token: the one-token certification gap.
* ``certify_before_release`` — host observes the score then emit (strict sync).
  Correct, but forces a device-host sync every step.
* ``one_token_commit_buffer`` — hold the latest token until the next step (or
  EOS), then release it only if its score is safe. Correct with +1 token of
  release latency; the GPU can advance while the prior token waits.

The score sequence is fixed so the result is deterministic and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Policy(StrEnum):
    RELEASE_BEFORE_CERTIFY = "release_before_certify"
    CERTIFY_BEFORE_RELEASE = "certify_before_release"
    ONE_TOKEN_COMMIT_BUFFER = "one_token_commit_buffer"


# Deterministic decode: tokens 0..n-1 with known scores.
# Index 3 is the first unsafe token under threshold 0.5.
DEFAULT_SCORES: tuple[float, ...] = (0.05, 0.08, 0.12, 0.91, 0.07, 0.04)
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class StepOutcome:
    step: int
    token_id: int
    score: float
    unsafe: bool
    released_this_step: tuple[int, ...]
    stopped: bool
    note: str = ""


@dataclass
class CertificationResult:
    policy: Policy
    threshold: float
    scores: tuple[float, ...]
    released_tokens: list[int] = field(default_factory=list)
    blocked_token: int | None = None
    leaked_unsafe_tokens: list[int] = field(default_factory=list)
    steps: list[StepOutcome] = field(default_factory=list)

    @property
    def is_correct(self) -> bool:
        return not self.leaked_unsafe_tokens

    @property
    def gap_tokens(self) -> int:
        """Number of unsafe tokens that crossed the serving boundary."""
        return len(self.leaked_unsafe_tokens)


def _is_unsafe(score: float, threshold: float) -> bool:
    return score >= threshold


def run_certification_gap(
    policy: Policy | str,
    scores: tuple[float, ...] | list[float] = DEFAULT_SCORES,
    threshold: float = DEFAULT_THRESHOLD,
) -> CertificationResult:
    """Simulate decode+score+release under ``policy``.

    Tokens are ``range(len(scores))``. A token is unsafe when
    ``score >= threshold``. Correct policies never append an unsafe token
    to ``released_tokens``.
    """
    policy = Policy(policy)
    scores_t = tuple(float(s) for s in scores)
    result = CertificationResult(policy=policy, threshold=threshold, scores=scores_t)

    pending_token: int | None = None
    pending_score: float | None = None

    for step, score in enumerate(scores_t):
        token_id = step
        unsafe = _is_unsafe(score, threshold)
        released_this_step: list[int] = []
        stopped = False
        note = ""

        if policy is Policy.RELEASE_BEFORE_CERTIFY:
            # Emit before host certification. Later certification finds the gap.
            result.released_tokens.append(token_id)
            released_this_step.append(token_id)
            if unsafe:
                result.leaked_unsafe_tokens.append(token_id)
                result.blocked_token = token_id
                stopped = True
                note = "unsafe certified after release (gap)"
            else:
                note = "released pre-certification"

        elif policy is Policy.CERTIFY_BEFORE_RELEASE:
            if unsafe:
                result.blocked_token = token_id
                stopped = True
                note = "blocked pre-release after host score read"
            else:
                result.released_tokens.append(token_id)
                released_this_step.append(token_id)
                note = "certified then released"

        elif policy is Policy.ONE_TOKEN_COMMIT_BUFFER:
            # Commit the buffered token (host certification of the prior step),
            # then park the current token in the one-slot buffer.
            if pending_token is not None:
                assert pending_score is not None
                if _is_unsafe(pending_score, threshold):
                    result.blocked_token = pending_token
                    stopped = True
                    note = f"blocked buffered token {pending_token}"
                else:
                    result.released_tokens.append(pending_token)
                    released_this_step.append(pending_token)
                    note = f"committed buffered token {pending_token}"

            if not stopped:
                if unsafe:
                    result.blocked_token = token_id
                    stopped = True
                    pending_token = None
                    pending_score = None
                    note = (note + "; " if note else "") + (
                        f"token {token_id} unsafe — not buffered"
                    )
                else:
                    pending_token = token_id
                    pending_score = score
                    note = (note + "; " if note else "") + f"buffered token {token_id}"

        result.steps.append(
            StepOutcome(
                step=step,
                token_id=token_id,
                score=score,
                unsafe=unsafe,
                released_this_step=tuple(released_this_step),
                stopped=stopped,
                note=note,
            )
        )
        if stopped:
            break

    # EOS: flush the commit buffer.
    if (
        policy is Policy.ONE_TOKEN_COMMIT_BUFFER
        and pending_token is not None
        and result.blocked_token is None
    ):
        assert pending_score is not None
        if _is_unsafe(pending_score, threshold):
            result.blocked_token = pending_token
        else:
            result.released_tokens.append(pending_token)
            result.steps.append(
                StepOutcome(
                    step=len(scores_t),
                    token_id=pending_token,
                    score=pending_score,
                    unsafe=False,
                    released_this_step=(pending_token,),
                    stopped=False,
                    note="flush commit buffer at EOS",
                )
            )

    return result


def compare_policies(
    scores: tuple[float, ...] | list[float] = DEFAULT_SCORES,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, CertificationResult]:
    return {p.value: run_certification_gap(p, scores=scores, threshold=threshold) for p in Policy}
