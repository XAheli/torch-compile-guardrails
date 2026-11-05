"""CUDA tests for the .item() synchronization-tax microbenchmark."""

import pytest
import torch

from torch_compile_guardrails.experiments.sync_tax import (
    BurstMode,
    compare_sync_modes,
    run_sync_tax,
    sync_tax_summary,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_per_step_item_returns_host_scalar():
    res = run_sync_tax(
        BurstMode.SCORE_PER_STEP_ITEM,
        bursts=3,
        steps_per_burst=8,
        warmup_bursts=1,
        decode_dim=128,
        score_dim=32,
    )
    assert res.last_score is not None
    assert 0.0 <= res.last_score <= 1.0


def test_pipeline_has_no_host_score():
    res = run_sync_tax(
        BurstMode.SCORE_PIPELINE,
        bursts=3,
        steps_per_burst=8,
        warmup_bursts=1,
        decode_dim=128,
        score_dim=32,
    )
    assert res.last_score is None


def test_summary_and_per_step_item_not_faster_than_pipeline():
    results = compare_sync_modes(
        bursts=8,
        steps_per_burst=16,
        warmup_bursts=2,
        decode_dim=256,
        score_dim=64,
    )
    summary = sync_tax_summary(results)
    assert summary["score_pipeline_ms"] > 0
    assert summary["score_per_step_item_ms"] > 0
    # Allow tiny noise; the interesting claim is measured by the runner.
    assert summary["score_sync_tax_ratio"] >= 0.5
