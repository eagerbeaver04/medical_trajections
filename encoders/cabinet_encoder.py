from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class CabinetEncoder(nn.Module):
    """
    Encodes cabinet token IDs via one-hot + MLP into d_model vector.

    Input:
        x: LongTensor of shape [batch, seq_len] or [batch]

    Output:
        Tensor of shape [batch, seq_len, d_model]
        or [batch, d_model]
    """

    def __init__(
        self,
        num_cabinets: int,
        d_model: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_cabinets <= 0:
            raise ValueError("num_cabinets must be > 0")

        hidden_dim = hidden_dim or d_model

        self.num_cabinets = num_cabinets
        self.d_model = d_model

        self.mlp = nn.Sequential(
            nn.Linear(num_cabinets, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: #(LongTensor)?
        if x.dim() not in (1, 2):
            raise ValueError(
                f"CabinetEncoder expects input of shape [B] or [B, T], got {tuple(x.shape)}"
            )

        original_shape_1d = False
        if x.dim() == 1:
            x = x.unsqueeze(1)  # [B, 1]
            original_shape_1d = True

        one_hot = F.one_hot(x, num_classes=self.num_cabinets).float()  # [B, T, C]
        out = self.mlp(one_hot)                                         # [B, T, d_model]

        if original_shape_1d:
            out = out.squeeze(1)  # [B, d_model]

        return out