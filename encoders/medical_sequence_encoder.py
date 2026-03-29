from __future__ import annotations

from typing import Sequence
import torch.nn as nn

from .cabinet_encoder import CabinetEncoder
from .condition_encoder import ConditionEncoder


class MedicalTokenEncoders(nn.Module):
    def __init__(
        self,
        feature_cardinalities: Sequence[int],
        num_cabinets: int,
        d_model: int,
        condition_feature_embedding_dim: int = 8,
        cabinet_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.condition_encoder = ConditionEncoder(
            feature_cardinalities=feature_cardinalities,
            feature_embedding_dim=condition_feature_embedding_dim,
            d_model=d_model,
            dropout=dropout,
        )
        self.cabinet_encoder = CabinetEncoder(
            num_cabinets=num_cabinets,
            d_model=d_model,
            hidden_dim=cabinet_hidden_dim,
            dropout=dropout,
        )