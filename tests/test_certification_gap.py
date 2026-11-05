"""Unit tests for the deterministic certification-gap simulator."""

from torch_compile_guardrails.experiments.certification_gap import (
    DEFAULT_SCORES,
    DEFAULT_THRESHOLD,
    Policy,
    compare_policies,
    run_certification_gap,
)


def test_naive_policy_leaks_exactly_one_unsafe_token():
    res = run_certification_gap(Policy.RELEASE_BEFORE_CERTIFY)
    assert res.leaked_unsafe_tokens == [3]
    assert res.gap_tokens == 1
    assert res.blocked_token == 3
    assert 3 in res.released_tokens
    assert res.released_tokens == [0, 1, 2, 3]
    assert not res.is_correct


def test_strict_sync_never_leaks():
    res = run_certification_gap(Policy.CERTIFY_BEFORE_RELEASE)
    assert res.leaked_unsafe_tokens == []
    assert res.is_correct
    assert res.blocked_token == 3
    assert res.released_tokens == [0, 1, 2]


def test_commit_buffer_never_leaks_and_delays_one_token():
    res = run_certification_gap(Policy.ONE_TOKEN_COMMIT_BUFFER)
    assert res.leaked_unsafe_tokens == []
    assert res.is_correct
    assert res.blocked_token == 3
    # Tokens 0,1,2 are safe and get committed; 3 is unsafe and never released.
    assert res.released_tokens == [0, 1, 2]
    # Step 0 only buffers — nothing released yet (one-token latency).
    assert res.steps[0].released_this_step == ()
    assert res.steps[1].released_this_step == (0,)


def test_compare_policies_covers_all():
    results = compare_policies(scores=DEFAULT_SCORES, threshold=DEFAULT_THRESHOLD)
    assert set(results) == {p.value for p in Policy}
    assert results["release_before_certify"].gap_tokens == 1
    assert results["certify_before_release"].gap_tokens == 0
    assert results["one_token_commit_buffer"].gap_tokens == 0


def test_all_safe_flush_buffer_at_eos():
    scores = (0.1, 0.2, 0.3)
    res = run_certification_gap(Policy.ONE_TOKEN_COMMIT_BUFFER, scores=scores)
    assert res.released_tokens == [0, 1, 2]
    assert res.blocked_token is None
    assert res.steps[-1].note == "flush commit buffer at EOS"
