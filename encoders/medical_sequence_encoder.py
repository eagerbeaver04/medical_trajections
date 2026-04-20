from __future__ import annotations

from typing import Sequence
import torch.nn as nn

from .cabinet_encoder import CabinetSimpleEncoder, CabinetMLPEncoder
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
        self.cabinet_encoder = CabinetSimpleEncoder(
            num_cabinets=num_cabinets, # Dictionary size
            d_model=d_model,
            padding_idx=0,
            # hidden_dim=cabinet_hidden_dim,
            # dropout=dropout,
        )