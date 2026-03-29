from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionEncoder(nn.Module):
    """
    Encodes structured categorical patient condition into a single d_model vector.

    Input:
        x: LongTensor of shape [batch, seq_len, num_features]
           or [batch, num_features]

    Output:
        Tensor of shape [batch, seq_len, d_model]
        or [batch, d_model]
    """

    def __init__(
        self,
        feature_cardinalities: Sequence[int],
        feature_embedding_dim: int,
        d_model: int,
        padding_idx: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.feature_cardinalities = list(feature_cardinalities)
        self.num_features = len(self.feature_cardinalities)
        self.feature_embedding_dim = feature_embedding_dim
        self.d_model = d_model

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=cardinality,
                    embedding_dim=feature_embedding_dim,
                    padding_idx=padding_idx,
                )
                for cardinality in self.feature_cardinalities
            ]
        )

        self.proj = nn.Sequential(
            nn.Linear(self.num_features * feature_embedding_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: #(LongTensor?)
        if x.dim() not in (2, 3):
            raise ValueError(
                f"ConditionEncoder expects input of shape [B, F] or [B, T, F], got {tuple(x.shape)}"
            )

        original_shape_2d = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, F]
            original_shape_2d = True

        batch_size, seq_len, num_features = x.shape
        if num_features != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} condition features, got {num_features}"
            )

        embedded_features = []
        for i, emb in enumerate(self.embeddings):
            feature_values = x[..., i]           # [B, T]
            feature_emb = emb(feature_values)    # [B, T, E]
            embedded_features.append(feature_emb)

        concat_emb = torch.cat(embedded_features, dim=-1)  # [B, T, F*E]
        out = self.proj(concat_emb)                        # [B, T, d_model]

        if original_shape_2d:
            out = out.squeeze(1)  # [B, d_model]

        return out