"""Module wrapper that owns a linear safety probe."""

from __future__ import annotations

import torch
from torch import nn

from .api import ScoringBackend, activation_score


class SafetyScorer(nn.Module):
    """A frozen linear probe with a selectable execution backend."""

    def __init__(
        self,
        hidden_size: int,
        *,
        backend: ScoringBackend | str = ScoringBackend.NATIVE,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.backend = ScoringBackend(backend)
        self.weight = nn.Parameter(torch.empty(hidden_size), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(()), requires_grad=False)
        nn.init.normal_(self.weight, std=hidden_size**-0.5)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return activation_score(activations, self.weight, self.bias, self.backend)
