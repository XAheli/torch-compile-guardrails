"""Numerical and compiler-contract tests for scoring backends."""

import pytest
import torch

from torch_compile_guardrails.scoring import ScoringBackend, activation_score
from torch_compile_guardrails.scoring.custom_op import custom_op_score


def _inputs(device: str, dtype: torch.dtype = torch.float32):
    generator = torch.Generator(device=device).manual_seed(17)
    activations = torch.randn(3, 5, 37, device=device, dtype=dtype, generator=generator)
    weight = torch.randn(37, device=device, dtype=dtype, generator=generator)
    bias = torch.randn((), device=device, dtype=dtype, generator=generator)
    return activations, weight, bias


@pytest.mark.parametrize("backend", list(ScoringBackend))
def test_backends_match_native_on_cpu(backend):
    inputs = _inputs("cpu")
    expected = activation_score(*inputs, ScoringBackend.NATIVE)
    actual = activation_score(*inputs, backend)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (3,)
    assert actual.dtype == torch.float32


def test_opaque_custom_op_contract_passes_opcheck():
    torch.library.opcheck(custom_op_score, _inputs("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("backend", list(ScoringBackend))
def test_cuda_backends_match_native(backend, dtype):
    inputs = _inputs("cuda", dtype)
    expected = activation_score(*inputs, ScoringBackend.NATIVE)
    actual = activation_score(*inputs, backend)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", list(ScoringBackend))
def test_fullgraph_compile_matches_eager(backend):
    inputs = _inputs("cuda", torch.bfloat16)

    def score(activations, weight, bias):
        return activation_score(activations, weight, bias, backend)

    expected = score(*inputs)
    compiled = torch.compile(score, fullgraph=True)
    actual = compiled(*inputs)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
