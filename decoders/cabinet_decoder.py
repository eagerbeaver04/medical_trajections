from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class CabinetDecoder(nn.Module):
    """
    Decodes hidden states into cabinet token logits.

    Input:
        x: Tensor [B, T, D] or [B, D]

    Output:
        Tensor [B, T, num_cabinets]
        or [B, num_cabinets]
    """

    def __init__(
        self,
        d_model: int,
        num_cabinets: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_cabinets <= 0:
            raise ValueError("num_cabinets must be > 0")

        hidden_dim = hidden_dim or d_model

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_cabinets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() not in (2, 3):
            raise ValueError(
                f"CabinetDecoder expects [B, D] or [B, T, D], got {tuple(x.shape)}"
            )

        original_2d = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            original_2d = True

        out = self.net(x)

        if original_2d:
            out = out.squeeze(1)

        return out