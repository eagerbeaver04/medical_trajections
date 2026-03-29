from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ConditionDecoder(nn.Module):
    """
    Decodes hidden states into structured condition feature logits.

    Input:
        x: Tensor [B, T, D] or [B, D]

    Output:
        list[Tensor]:
            one tensor per feature
            each tensor has shape [B, T, cardinality]
            or [B, cardinality] if input was [B, D]
    """

    def __init__(
        self,
        d_model: int,
        feature_cardinalities: Sequence[int],
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or d_model
        self.feature_cardinalities = list(feature_cardinalities)

        self.shared = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, cardinality) for cardinality in self.feature_cardinalities]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.dim() not in (2, 3):
            raise ValueError(
                f"ConditionDecoder expects [B, D] or [B, T, D], got {tuple(x.shape)}"
            )

        original_2d = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            original_2d = True

        h = self.shared(x)
        outputs = [head(h) for head in self.heads]

        if original_2d:
            outputs = [out.squeeze(1) for out in outputs]

        return outputs