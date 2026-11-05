"""Tests for model- and layer-bound probe artifacts."""

import pytest
import torch

from torch_compile_guardrails.models import GuardedQwen3ForCausalLM
from torch_compile_guardrails.scoring import ProbeMetadata, load_probe, save_probe

from .test_qwen3 import _tiny_qwen3


def test_probe_round_trip_and_model_loading(tmp_path):
    path = tmp_path / "probe.pt"
    weight = torch.linspace(-1, 1, 64)
    bias = torch.tensor(0.25)
    metadata = ProbeMetadata(
        model_id="",
        score_layer=0,
        hidden_size=64,
        dataset="unit-test",
    )
    save_probe(path, weight, bias, metadata)
    loaded_weight, loaded_bias, loaded_metadata = load_probe(
        path, expected_hidden_size=64, expected_layer=0, expected_model_id=None
    )
    torch.testing.assert_close(loaded_weight, weight)
    torch.testing.assert_close(loaded_bias, bias)
    assert loaded_metadata == metadata

    guarded = GuardedQwen3ForCausalLM(_tiny_qwen3(), score_layer=0, probe_path=path)
    assert guarded.scorer is not None
    torch.testing.assert_close(guarded.scorer.weight, weight)
    torch.testing.assert_close(guarded.scorer.bias, bias)


def test_probe_rejects_wrong_layer(tmp_path):
    path = tmp_path / "probe.pt"
    save_probe(
        path,
        torch.zeros(64),
        torch.zeros(()),
        ProbeMetadata(model_id="tiny", score_layer=1, hidden_size=64, dataset="unit-test"),
    )
    with pytest.raises(ValueError, match="probe layer"):
        load_probe(path, expected_hidden_size=64, expected_layer=0)
