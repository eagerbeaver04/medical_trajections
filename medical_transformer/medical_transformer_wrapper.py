from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        _, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, T]
        return x + self.embedding(positions)


def interleave_tokens(
    condition_tokens: torch.Tensor,  # [B, Tc, D]
    cabinet_tokens: torch.Tensor,    # [B, Tk, D]
) -> torch.Tensor:
    B, Tc, D = condition_tokens.shape
    _, Tk, D2 = cabinet_tokens.shape

    if D != D2:
        raise ValueError(f"Embedding dims differ: {D} vs {D2}")

    if Tc != Tk + 1:
        raise ValueError(
            f"Expected number of condition tokens Tc == Tk + 1, got Tc={Tc}, Tk={Tk}"
        )

    out = torch.empty(
        B,
        Tc + Tk,
        D,
        device=condition_tokens.device,
        dtype=condition_tokens.dtype,
    )
    out[:, 0::2, :] = condition_tokens
    out[:, 1::2, :] = cabinet_tokens
    return out


class TransformerWrapper(nn.Module):
    """
    Wraps:
      - your imported MedicalTokenEncoders
      - a TransformerEncoder
      - your imported MedicalOutputHeads

    Sequence convention:
      c0, a0, c1, a1, ..., a_{T-1}, cT

    Shapes:
      conditions:   [B, Tc, F]
      cabinets:     [B, Tk]
      padding_mask: [B, T_total] where T_total = Tc + Tk and True means PAD

    Required relation:
      Tc == Tk + 1
    """

    def __init__(
        self,
        token_encoders: nn.Module,
        transformer: nn.Module,
        output_heads: nn.Module,
        d_model: int,
        max_seq_len: int,
        use_causal_mask: bool = True,
    ) -> None:
        super().__init__()
        self.token_encoders = token_encoders
        self.transformer = transformer
        self.output_heads = output_heads
        self.position = LearnedPositionalEncoding(max_seq_len, d_model)
        self.use_causal_mask = use_causal_mask

    @staticmethod
    def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        # True = blocked
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def encode_tokens(
        self,
        conditions: torch.Tensor,  # [B, Tc, F]
        cabinets: torch.Tensor,    # [B, Tk]
    ) -> torch.Tensor:
        condition_tokens = self.token_encoders.condition_encoder(conditions)  # [B, Tc, D]
        cabinet_tokens = self.token_encoders.cabinet_encoder(cabinets)        # [B, Tk, D]

        x = interleave_tokens(condition_tokens, cabinet_tokens)               # [B, T, D]
        x = self.position(x)
        return x

    def forward_hidden(
        self,
        conditions: torch.Tensor,
        cabinets: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.encode_tokens(conditions, cabinets)  # [B, T, D]
        seq_len = x.size(1)

        causal_mask = None
        if self.use_causal_mask:
            causal_mask = self.build_causal_mask(seq_len, x.device)

        out = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return out  # [B, T, D]

    def forward_next_cabinet(
        self,
        conditions: torch.Tensor,
        cabinets: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict cabinet at each cabinet step from condition-token positions.

        Returns:
          cabinet_logits: [B, Tk, num_cabinets]
        """
        hidden = self.forward_hidden(
            conditions=conditions,
            cabinets=cabinets,
            padding_mask=padding_mask,
        )  # [B, T, D]

        # condition positions: 0,2,4,...,2*Tk
        # for next cabinet prediction use all condition positions except final cT
        hidden_for_cabinet = hidden[:, 0:-1:2, :]  # [B, Tk, D]

        cabinet_logits = self.output_heads.cabinet_decoder(hidden_for_cabinet)
        return cabinet_logits

    def forward_next_condition(
        self,
        conditions: torch.Tensor,
        cabinets: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        """
        Predict next condition from cabinet-token positions.

        Returns:
          list of feature logits, each of shape [B, Tk, cardinality_i]
        """
        hidden = self.forward_hidden(
            conditions=conditions,
            cabinets=cabinets,
            padding_mask=padding_mask,
        )  # [B, T, D]

        # cabinet positions: 1,3,5,...
        hidden_for_condition = hidden[:, 1::2, :]  # [B, Tk, D]

        condition_logits = self.output_heads.condition_decoder(hidden_for_condition)
        return condition_logits

    def forward(
        self,
        conditions: torch.Tensor,
        cabinets: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, object]:
        cabinet_logits = self.forward_next_cabinet(
            conditions=conditions,
            cabinets=cabinets,
            padding_mask=padding_mask,
        )

        condition_logits = self.forward_next_condition(
            conditions=conditions,
            cabinets=cabinets,
            padding_mask=padding_mask,
        )

        return {
            "cabinet_logits": cabinet_logits,
            "condition_logits": condition_logits,
        }

    def compute_cabinet_loss(
        self,
        cabinet_logits: torch.Tensor,  # [B, Tk, V]
        cabinet_targets: torch.Tensor, # [B, Tk]
        cabinet_mask: torch.Tensor,    # [B, Tk] bool, True for valid targets
    ) -> torch.Tensor:
        B, Tk, V = cabinet_logits.shape

        loss = F.cross_entropy(
            cabinet_logits.reshape(B * Tk, V),
            cabinet_targets.reshape(B * Tk),
            reduction="none",
        ).reshape(B, Tk)

        cabinet_mask = cabinet_mask.bool()
        if cabinet_mask.sum() == 0:
            return loss.mean() * 0.0

        return loss[cabinet_mask].mean()

    def compute_condition_loss(
        self,
        condition_logits: list[torch.Tensor],
        condition_targets: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> torch.Tensor:
        condition_mask = condition_mask.bool()
        total_loss = 0.0
        total_count = 0

        for feature_idx, logits_i in enumerate(condition_logits):
            targets_i = condition_targets[:, :, feature_idx]
            B, Tk, Ci = logits_i.shape
            
            loss_i = F.cross_entropy(
                logits_i.reshape(B * Tk, Ci),
                targets_i.reshape(B * Tk),
                reduction="none",
            ).reshape(B, Tk)
            
            masked_loss = loss_i[condition_mask]  # [valid_count]
            total_loss += masked_loss.sum()
            total_count += masked_loss.numel()

        if total_count == 0:
            # Пустая маска: возвращаем нулевой тензор, связанный со всеми головами
            return sum(logits.sum() for logits in condition_logits) * 0.0

        return total_loss / total_count

    def compute_losses(
        self,
        batch: dict[str, torch.Tensor],
        cabinet_loss_weight: float = 1.0,
        condition_loss_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Expects batch keys:
          conditions:      [B, Tc, F]
          cabinets:        [B, Tk]
          padding_mask:    [B, T_total]
          cabinet_mask:    [B, Tk]
          condition_mask:  [B, Tc]

        Target convention:
          next-cabinet targets = cabinets
          next-condition targets = conditions[:, 1:, :]
        """
        outputs = self.forward(
            conditions=batch["conditions"],
            cabinets=batch["cabinets"],
            padding_mask=batch.get("padding_mask"),
        )

        cabinet_logits = outputs["cabinet_logits"]      # [B, Tk, V]
        condition_logits = outputs["condition_logits"]  # list[[B, Tk, Ci]]

        cabinet_targets = batch["cabinets"]             # [B, Tk]
        cabinet_mask = batch["cabinet_mask"]            # [B, Tk]

        condition_targets = batch["conditions"][:, 1:, :]   # [B, Tk, F]
        condition_mask = batch["condition_mask"][:, 1:]     # [B, Tk]

        cabinet_loss = self.compute_cabinet_loss(
            cabinet_logits=cabinet_logits,
            cabinet_targets=cabinet_targets,
            cabinet_mask=cabinet_mask,
        )

        condition_loss = self.compute_condition_loss(
            condition_logits=condition_logits,
            condition_targets=condition_targets,
            condition_mask=condition_mask,
        )

        total_loss = (
            cabinet_loss_weight * cabinet_loss
            + condition_loss_weight * condition_loss
        )

        return {
            "loss": total_loss,
            "cabinet_loss": cabinet_loss,
            "condition_loss": condition_loss,
            "cabinet_logits": cabinet_logits,
            "condition_logits": condition_logits,
        }