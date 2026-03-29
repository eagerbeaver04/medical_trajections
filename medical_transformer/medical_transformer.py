from __future__ import annotations

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len] # register buffer
        return self.dropout(x)


class MedicalTransformer(nn.Module):
    """
    Wrapper around existing encoders/decoders.

    Sequence structure:
        C0, K0, C1, K1, ..., K(T-1), C(T)

    Inputs:
        conditions: [B, T+1, F]
        cabinets:   [B, T]

    Outputs:
        cabinet_logits: predicted from condition positions [B, T, num_cabinets]
        next_condition_logits: list of feature logits, each [B, T, cardinality_i]
    """

    def __init__(
        self,
        condition_encoder: nn.Module,
        cabinet_encoder: nn.Module,
        condition_decoder: nn.Module,
        cabinet_decoder: nn.Module,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        max_len: int = 4096,
        norm_first: bool = True,
    ) -> None:
        super().__init__()

        self.condition_encoder = condition_encoder
        self.cabinet_encoder = cabinet_encoder
        self.condition_decoder = condition_decoder
        self.cabinet_decoder = cabinet_decoder

        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

    @staticmethod
    def interleave_tokens(
        condition_emb: torch.Tensor,  # [B, T+1, D]
        cabinet_emb: torch.Tensor,    # [B, T, D]
    ) -> torch.Tensor:
        bsz, t_cond, d_model = condition_emb.shape
        bsz2, t_cab, d_model2 = cabinet_emb.shape

        if bsz != bsz2 or d_model != d_model2:
            raise ValueError("Condition and cabinet embeddings must match in batch and d_model")
        if t_cond != t_cab + 1:
            raise ValueError(
                f"Expected number of conditions = number of cabinets + 1, got {t_cond} and {t_cab}"
            )

        total_len = t_cond + t_cab  # 2T + 1
        out = condition_emb.new_empty(bsz, total_len, d_model)

        out[:, 0::2, :] = condition_emb
        out[:, 1::2, :] = cabinet_emb

        return out

    @staticmethod
    def split_interleaved(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            condition_hidden: [B, T+1, D]
            cabinet_hidden:   [B, T, D]
        """
        condition_hidden = hidden[:, 0::2, :]
        cabinet_hidden = hidden[:, 1::2, :]
        return condition_hidden, cabinet_hidden

    @staticmethod
    def build_autoregressive_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Causal self-attention mask for TransformerEncoder.
        Shape: [seq_len, seq_len]
        True means blocked.
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(
        self,
        conditions: torch.LongTensor,   # [B, T+1, F]
        cabinets: torch.LongTensor,     # [B, T]
        padding_mask: torch.Tensor | None = None,   # [B, 2T+1], True means PAD
        causal: bool = True,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        # 1. Encode raw inputs into dense token embeddings
        condition_emb = self.condition_encoder(conditions)  # [B, T+1, D]
        cabinet_emb = self.cabinet_encoder(cabinets)        # [B, T, D]

        # 2. Interleave: C0, K0, C1, K1, ..., CT
        x = self.interleave_tokens(condition_emb, cabinet_emb)  # [B, 2T+1, D]

        # 3. Add positional encoding
        x = self.positional_encoding(x)

        # 4. Transformer
        attn_mask = None
        if causal:
            attn_mask = self.build_autoregressive_mask(
                seq_len=x.size(1),
                device=x.device,
            )

        hidden = self.transformer(
            x,
            mask=attn_mask,
            src_key_padding_mask=padding_mask,
        )  # [B, 2T+1, D]

        # 5. Split by token type
        condition_hidden, cabinet_hidden = self.split_interleaved(hidden)
        # condition_hidden: [B, T+1, D]
        # cabinet_hidden:   [B, T, D]

        # 6. Decode according to sequence semantics
        # From C_t predict K_t => use all condition positions except the last
        cabinet_logits = self.cabinet_decoder(condition_hidden[:, :-1, :])  # [B, T, ...]

        # From K_t predict C_{t+1} => use all cabinet positions
        next_condition_logits = self.condition_decoder(cabinet_hidden)       # list of [B, T, ...]

        return {
            "hidden": hidden,
            "condition_hidden": condition_hidden,
            "cabinet_hidden": cabinet_hidden,
            "cabinet_logits": cabinet_logits,
            "next_condition_logits": next_condition_logits,
        }