"""Reference-equivalence tests for the explicit Qwen3 activation tap."""

import pytest
import torch
from transformers.cache_utils import StaticCache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from torch_compile_guardrails.models import GuardedQwen3ForCausalLM
from torch_compile_guardrails.scoring import ScoringBackend


def _tiny_qwen3(device: str = "cpu") -> Qwen3ForCausalLM:
    torch.manual_seed(11)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        layer_types=["full_attention", "full_attention"],
        use_cache=False,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval().to(device)


@pytest.mark.parametrize("score_layer", [0, 1])
def test_guarded_prefill_logits_match_hugging_face(score_layer):
    model = _tiny_qwen3()
    guarded = GuardedQwen3ForCausalLM(model, score_layer=score_layer)
    input_ids = torch.tensor([[1, 7, 9, 4], [3, 2, 6, 5]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        expected = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
        ).logits
        logits, scores = guarded(input_ids, attention_mask)

    torch.testing.assert_close(logits, expected)
    assert logits.shape == (2, 1, 128)
    assert scores.shape == (2,)
    assert torch.all((scores >= 0.0) & (scores <= 1.0))


def test_invalid_score_layer_is_rejected():
    model = _tiny_qwen3()
    with pytest.raises(ValueError, match="score_layer"):
        GuardedQwen3ForCausalLM(model, score_layer=2)


def test_disabled_scorer_preserves_logits_and_returns_zero_sentinel():
    model = _tiny_qwen3()
    guarded = GuardedQwen3ForCausalLM(model, score_layer=None)
    input_ids = torch.tensor([[1, 7, 9, 4]])
    with torch.no_grad():
        expected = model(input_ids=input_ids, use_cache=False, logits_to_keep=1).logits
        logits, score = guarded(input_ids)
    torch.testing.assert_close(logits, expected)
    torch.testing.assert_close(score, torch.zeros(1))


def test_cached_step_matches_full_prefix_logits_and_score():
    model = _tiny_qwen3()
    torch.manual_seed(3)
    guarded = GuardedQwen3ForCausalLM(model, score_layer=0)
    prompt = torch.tensor([[1, 7, 9, 4]])
    consumed_token = torch.tensor([[12]])

    with torch.no_grad():
        _, _, cache = guarded.forward_with_cache(prompt)
        cached_logits, cached_score, _ = guarded.forward_with_cache(
            consumed_token, past_key_values=cache
        )
        full_logits, full_score = guarded(torch.cat((prompt, consumed_token), dim=1))

    torch.testing.assert_close(cached_logits, full_logits, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(cached_score, full_score, rtol=1e-5, atol=1e-5)


def test_static_cache_matches_dynamic_cache():
    model = _tiny_qwen3()
    torch.manual_seed(3)
    guarded = GuardedQwen3ForCausalLM(model, score_layer=0)
    prompt = torch.tensor([[1, 7, 9, 4]])
    token = torch.tensor([[12]])
    static_cache = StaticCache(config=model.config, max_cache_len=8)

    with torch.no_grad():
        _, _, dynamic_cache = guarded.forward_with_cache(prompt)
        dynamic_logits, dynamic_score, _ = guarded.forward_with_cache(
            token, past_key_values=dynamic_cache
        )
        _, _, static_cache = guarded.forward_with_cache(prompt, past_key_values=static_cache)
        static_logits, static_score, _ = guarded.forward_with_cache(
            token, past_key_values=static_cache
        )

    torch.testing.assert_close(static_logits, dynamic_logits, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(static_score, dynamic_score, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", list(ScoringBackend))
def test_guarded_prefill_compiles_fullgraph_on_cuda(backend):
    model = _tiny_qwen3("cuda").to(torch.bfloat16)
    guarded = GuardedQwen3ForCausalLM(model, score_layer=0, scoring_backend=backend).eval()
    input_ids = torch.tensor([[1, 7, 9, 4]], device="cuda")
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        expected = guarded(input_ids, attention_mask)
        compiled = torch.compile(guarded, fullgraph=True)
        actual = compiled(input_ids, attention_mask)

    torch.testing.assert_close(actual[0], expected[0], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cached_decode_step_compiles_fullgraph_on_cuda():
    model = _tiny_qwen3("cuda").to(torch.bfloat16)
    guarded = GuardedQwen3ForCausalLM(model, score_layer=0).eval()
    prompt = torch.tensor([[1, 7, 9]], device="cuda")
    token = torch.tensor([[4]], device="cuda")
    with torch.no_grad():
        _, _, cache = guarded.forward_with_cache(prompt)
        compiled = torch.compile(guarded.forward_with_cache, fullgraph=True)
        logits, score, updated_cache = compiled(token, None, cache)
    assert logits.shape == (1, 1, 128)
    assert score.shape == (1,)
    assert updated_cache.get_seq_length() == 4
