"""Measure device-host synchronization when reading a CUDA safety score.

Timing each iteration with a synchronizing CUDA event would hide the cost under
study. Each sample therefore measures host wall time over a burst: pipeline
modes enqueue all work and synchronize once, while item modes read a scalar on
every step. This synthetic benchmark complements the real-model benchmark.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from enum import StrEnum

import torch


class BurstMode(StrEnum):
    DECODE_PIPELINE = "decode_pipeline"
    DECODE_PER_STEP_ITEM = "decode_per_step_item"
    SCORE_PIPELINE = "score_pipeline"
    SCORE_PER_STEP_ITEM = "score_per_step_item"


@dataclass(frozen=True)
class SyncTaxResult:
    mode: BurstMode
    bursts: int
    steps_per_burst: int
    decode_dim: int
    score_dim: int
    median_ms_per_step: float
    mean_ms_per_step: float
    p90_ms_per_step: float
    last_score: float | None

    def as_dict(self) -> dict[str, float | int | str | None]:
        return {
            "mode": self.mode.value,
            "bursts": self.bursts,
            "steps_per_burst": self.steps_per_burst,
            "decode_dim": self.decode_dim,
            "score_dim": self.score_dim,
            "median_ms_per_step": self.median_ms_per_step,
            "mean_ms_per_step": self.mean_ms_per_step,
            "p90_ms_per_step": self.p90_ms_per_step,
            "last_score": self.last_score,
        }


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the sync-tax experiment")
    return torch.device("cuda")


def _make_tensors(
    device: torch.device, decode_dim: int, score_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(0)
    decode_w = torch.randn(decode_dim, decode_dim, device=device, generator=generator)
    decode_x = torch.randn(decode_dim, decode_dim, device=device, generator=generator)
    score_w = torch.randn(score_dim, 1, device=device, generator=generator)
    activation = torch.randn(score_dim, device=device, generator=generator)
    return decode_x, decode_w, score_w, activation


def _score(vector: torch.Tensor, score_w: torch.Tensor, score_dim: int) -> torch.Tensor:
    pooled = vector.reshape(-1)[:score_dim]
    return torch.sigmoid(pooled @ score_w).squeeze()


def run_sync_tax(
    mode: BurstMode | str,
    *,
    bursts: int = 50,
    steps_per_burst: int = 64,
    warmup_bursts: int = 5,
    decode_dim: int = 1024,
    score_dim: int = 256,
) -> SyncTaxResult:
    mode = BurstMode(mode)
    if bursts <= 0 or steps_per_burst <= 0 or warmup_bursts < 0:
        raise ValueError("bursts and steps must be positive; warmup must be non-negative")
    device = _require_cuda()
    decode_x, decode_w, score_w, activation = _make_tensors(device, decode_dim, score_dim)
    last_score: float | None = None
    per_step_ms: list[float] = []

    def decode_score() -> torch.Tensor:
        return _score(decode_x @ decode_w, score_w, score_dim)

    def score_only() -> torch.Tensor:
        return _score(activation, score_w, score_dim)

    score_function = (
        decode_score
        if mode in (BurstMode.DECODE_PIPELINE, BurstMode.DECODE_PER_STEP_ITEM)
        else score_only
    )
    reads_each_step = mode in (
        BurstMode.DECODE_PER_STEP_ITEM,
        BurstMode.SCORE_PER_STEP_ITEM,
    )

    for burst_index in range(warmup_bursts + bursts):
        torch.cuda.synchronize()
        start = time.perf_counter()
        if reads_each_step:
            for _ in range(steps_per_burst):
                last_score = float(score_function().item())
        else:
            sink = torch.zeros((), device=device)
            for _ in range(steps_per_burst):
                sink = sink + score_function()
            torch.cuda.synchronize()
            last_score = None
        torch.cuda.synchronize()
        if burst_index >= warmup_bursts:
            per_step_ms.append((time.perf_counter() - start) * 1000.0 / steps_per_burst)

    ordered = sorted(per_step_ms)
    return SyncTaxResult(
        mode=mode,
        bursts=bursts,
        steps_per_burst=steps_per_burst,
        decode_dim=decode_dim,
        score_dim=score_dim,
        median_ms_per_step=statistics.median(per_step_ms),
        mean_ms_per_step=statistics.fmean(per_step_ms),
        p90_ms_per_step=ordered[round(0.9 * (len(ordered) - 1))],
        last_score=last_score,
    )


def compare_sync_modes(
    *,
    bursts: int = 50,
    steps_per_burst: int = 64,
    warmup_bursts: int = 5,
    decode_dim: int = 1024,
    score_dim: int = 256,
) -> dict[str, SyncTaxResult]:
    return {
        mode.value: run_sync_tax(
            mode,
            bursts=bursts,
            steps_per_burst=steps_per_burst,
            warmup_bursts=warmup_bursts,
            decode_dim=decode_dim,
            score_dim=score_dim,
        )
        for mode in BurstMode
    }


def sync_tax_summary(results: dict[str, SyncTaxResult]) -> dict[str, float]:
    decode_pipeline = results[BurstMode.DECODE_PIPELINE.value].median_ms_per_step
    decode_item = results[BurstMode.DECODE_PER_STEP_ITEM.value].median_ms_per_step
    score_pipeline = results[BurstMode.SCORE_PIPELINE.value].median_ms_per_step
    score_item = results[BurstMode.SCORE_PER_STEP_ITEM.value].median_ms_per_step
    return {
        "decode_pipeline_ms": decode_pipeline,
        "decode_per_step_item_ms": decode_item,
        "decode_sync_tax_ms": decode_item - decode_pipeline,
        "decode_sync_tax_ratio": decode_item / decode_pipeline,
        "score_pipeline_ms": score_pipeline,
        "score_per_step_item_ms": score_item,
        "score_sync_tax_ms": score_item - score_pipeline,
        "score_sync_tax_ratio": score_item / score_pipeline,
    }
