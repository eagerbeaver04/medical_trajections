from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from .cabinet_decoder import CabinetDecoder
from .condition_decoder import ConditionDecoder

class MedicalOutputHeads(nn.Module):
    def __init__(
        self,
        d_model: int,
        feature_cardinalities: Sequence[int],
        num_cabinets: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.condition_decoder = ConditionDecoder(
            d_model=d_model,
            feature_cardinalities=feature_cardinalities,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.cabinet_decoder = CabinetDecoder(
            d_model=d_model,
            num_cabinets=num_cabinets,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )