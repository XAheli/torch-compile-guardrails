"""Release-invariant tests for cache-aware Qwen decoding."""

import torch
from transformers import StaticCache

from torch_compile_guardrails.models import GuardedQwen3ForCausalLM
from torch_compile_guardrails.serve import (
    EnforcementPolicy,
    generate_strict,
    generate_with_commit_buffer,
)

from .test_qwen3 import _tiny_qwen3


def _guarded_model() -> GuardedQwen3ForCausalLM:
    torch.manual_seed(5)
    return GuardedQwen3ForCausalLM(_tiny_qwen3(), score_layer=0)


def test_commit_buffer_releases_only_verified_tokens():
    result = generate_with_commit_buffer(
        _guarded_model(), torch.tensor([[1, 7, 9]]), max_new_tokens=4, threshold=1.1
    )
    assert result.blocked_token_id is None
    assert len(result.released_token_ids) == 4
    assert result.released_token_ids == [v.token_id for v in result.verified_tokens]
    assert [v.decode_step for v in result.verified_tokens] == [0, 1, 2, 3]
    assert result.policy is EnforcementPolicy.COMMIT_BUFFER
    assert [v.absolute_position for v in result.verified_tokens] == [3, 4, 5, 6]
    assert all(
        v.score_covers_through_position == v.absolute_position for v in result.verified_tokens
    )


def test_commit_buffer_never_releases_blocked_token():
    result = generate_with_commit_buffer(
        _guarded_model(), torch.tensor([[1, 7, 9]]), max_new_tokens=4, threshold=0.0
    )
    assert result.released_token_ids == []
    assert result.blocked_token_id is not None
    assert result.verified_tokens[0].token_id == result.blocked_token_id
    assert result.blocked_token_id not in result.released_token_ids


def test_zero_token_request_does_not_run_decode():
    result = generate_with_commit_buffer(
        _guarded_model(), torch.tensor([[1, 7, 9]]), max_new_tokens=0, threshold=0.5
    )
    assert result.released_token_ids == []
    assert result.verified_tokens == []


def test_strict_quarantines_complete_safe_response_then_releases():
    result = generate_strict(
        _guarded_model(), torch.tensor([[1, 7, 9]]), max_new_tokens=4, threshold=1.1
    )
    assert result.policy is EnforcementPolicy.STRICT
    assert result.blocked_token_id is None
    assert result.released_token_ids == [token.token_id for token in result.verified_tokens]


def test_strict_releases_nothing_if_later_token_is_blocked():
    model = _guarded_model()
    calls = 0

    def block_second(input_ids, attention_mask, past_key_values):
        nonlocal calls
        logits, score, cache = model.forward_with_cache(input_ids, attention_mask, past_key_values)
        score = torch.zeros_like(score) if calls == 0 else torch.ones_like(score)
        calls += 1
        return logits, score, cache

    result = generate_strict(
        model,
        torch.tensor([[1, 7, 9]]),
        max_new_tokens=4,
        threshold=0.5,
        cached_forward=block_second,
    )
    assert len(result.verified_tokens) == 2
    assert result.blocked_token_id == result.verified_tokens[1].token_id
    assert result.released_token_ids == []


def test_compiled_callable_is_decode_only_not_prefill():
    model = _guarded_model()
    prompt = torch.tensor([[1, 7, 9]])
    cache = StaticCache(config=model.causal_lm.config, max_cache_len=8)
    observed_lengths: list[int] = []

    def recording_decode(input_ids, attention_mask, past_key_values):
        del attention_mask
        observed_lengths.append(input_ids.shape[1])
        assert past_key_values.layers[0].is_initialized
        return model.forward_with_cache(input_ids, None, past_key_values)

    generate_with_commit_buffer(
        model,
        prompt,
        max_new_tokens=2,
        threshold=1.1,
        cached_forward=recording_decode,
        initial_cache=cache,
    )
    assert observed_lengths == [1, 1]
