from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from structures.medical_sequence import MedicalSequence
from structures.patient_statuses import (
    PatientConditionSchema,
    PatientCondition,
    get_padding_conditions,
)
from structures.cabinets import ICabinet, PaddingCabinet
from .sequence_generator import SequenceGenerator


@dataclass
class MedicalSequenceRecord:
    sequence_id: str
    sequence: MedicalSequence
    terminal_status: Optional[Any] = None


class MedicalSequenceDataset(Dataset):
    def __init__(
        self,
        records: list[MedicalSequenceRecord],
        condition_schema: PatientConditionSchema,
        pad_cabinet_token_id: int = 0,
        pad_to_max_length: bool = True,
        encode_condition_pad_for_model: bool = True,
    ) -> None:
        self._records = records
        self._condition_schema = condition_schema
        self._pad_cabinet_token_id = pad_cabinet_token_id
        self._pad_to_max_length = pad_to_max_length
        self._encode_condition_pad_for_model = encode_condition_pad_for_model

        self._validate()

        self._max_conditions_len = max(
            (len(r.sequence.conditions_sequence) for r in self._records),
            default=0,
        )
        self._max_cabinets_len = max(
            (len(r.sequence.cabinet_sequence) for r in self._records),
            default=0,
        )

    @classmethod
    def from_generator(
        cls,
        generator: SequenceGenerator,
        n_sequences: int,
        max_steps: int = 50,
        pad_cabinet_token_id: int = 0,
        pad_to_max_length: bool = True,
        encode_condition_pad_for_model: bool = True,
    ) -> "MedicalSequenceDataset":
        records: list[MedicalSequenceRecord] = []

        for i in range(n_sequences):
            sequence, terminal_status = generator.generate_sequence(max_steps=max_steps)
            records.append(
                MedicalSequenceRecord(
                    sequence_id=f"seq_{i}",
                    sequence=sequence,
                    terminal_status=terminal_status,
                )
            )

        return cls(
            records=records,
            condition_schema=generator.schema,
            pad_cabinet_token_id=pad_cabinet_token_id,
            pad_to_max_length=pad_to_max_length,
            encode_condition_pad_for_model=encode_condition_pad_for_model,
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self._records[idx]

        conditions = record.sequence.conditions_sequence
        cabinets = record.sequence.cabinet_sequence

        if self._pad_to_max_length:
            padded_conditions = self._pad_conditions(conditions, self._max_conditions_len)
            padded_cabinets = self._pad_cabinets(cabinets, self._max_cabinets_len)
        else:
            padded_conditions = conditions
            padded_cabinets = cabinets

        condition_tensor = self._conditions_to_tensor(padded_conditions)
        cabinet_tensor = self._cabinets_to_tensor(padded_cabinets)

        condition_mask = self._build_condition_mask(len(conditions), len(padded_conditions))
        cabinet_mask = self._build_cabinet_mask(len(cabinets), len(padded_cabinets))
        padding_mask = self._build_interleaved_padding_mask(
            condition_mask=condition_mask,
            cabinet_mask=cabinet_mask,
        )

        return {
            "sequence_id": record.sequence_id,
            "conditions": condition_tensor,
            "cabinets": cabinet_tensor,
            "condition_mask": condition_mask,
            "cabinet_mask": cabinet_mask,
            "padding_mask": padding_mask,
            "terminal_status": record.terminal_status,
        }

    def _validate(self) -> None:
        seen_ids: set[str] = set()

        for record in self._records:
            if record.sequence_id in seen_ids:
                raise ValueError(f"Duplicate sequence_id: {record.sequence_id}")
            seen_ids.add(record.sequence_id)

            sequence = record.sequence

            if len(sequence.conditions_sequence) != len(sequence.cabinet_sequence) + 1:
                raise ValueError(
                    f"{record.sequence_id}: "
                    f"len(conditions_sequence) must equal len(cabinet_sequence) + 1"
                )

            for t, condition in enumerate(sequence.conditions_sequence):
                self._validate_condition(record.sequence_id, t, condition)

            for t, cabinet in enumerate(sequence.cabinet_sequence):
                self._validate_cabinet(record.sequence_id, t, cabinet)

    def _validate_condition(
        self,
        sequence_id: str,
        t: int,
        condition: PatientCondition,
    ) -> None:
        if len(condition) != self._condition_schema.patient_condition_len():
            raise ValueError(f"{sequence_id}: invalid condition length at step {t}")

        for i, cond in enumerate(condition.conditions):
            self._condition_schema.validate_number(i, cond.number, allow_pad=True)

    def _validate_cabinet(
        self,
        sequence_id: str,
        t: int,
        cabinet: ICabinet,
    ) -> None:
        if cabinet.token_id == self._pad_cabinet_token_id:
            return

        if cabinet.token_id < 0:
            raise ValueError(f"{sequence_id}: invalid cabinet token_id at step {t}")

    def _pad_conditions(
        self,
        conditions: list[PatientCondition],
        target_len: int,
    ) -> list[PatientCondition]:
        if len(conditions) > target_len:
            raise ValueError("conditions length exceeds target_len")

        out = list(conditions)
        while len(out) < target_len:
            out.append(get_padding_conditions(schema=self._condition_schema, time=0))
        return out

    def _pad_cabinets(
        self,
        cabinets: list[ICabinet],
        target_len: int,
    ) -> list[ICabinet]:
        if len(cabinets) > target_len:
            raise ValueError("cabinets length exceeds target_len")

        out = list(cabinets)
        while len(out) < target_len:
            out.append(PaddingCabinet())
        return out

    def _condition_number_to_model_id(self, feature_idx: int, value: int) -> int:
        """
        Converts raw condition value to embedding-safe id.

        Raw:
            valid states: 0 .. n_states-1
            pad: -1

        Model ids:
            valid states: 0 .. n_states-1
            pad: n_states
        """
        if value == self._condition_schema.PAD:
            return self._condition_schema.state_count(feature_idx)
        return value

    def _conditions_to_tensor(
        self,
        conditions: list[PatientCondition],
    ) -> torch.LongTensor:
        rows: list[list[int]] = []

        for condition in conditions:
            row: list[int] = []

            for feature_idx, value in enumerate(condition.as_numbers()):
                if self._encode_condition_pad_for_model:
                    mapped_value = self._condition_number_to_model_id(feature_idx, value)
                else:
                    mapped_value = value
                row.append(mapped_value)

            rows.append(row)

        return torch.tensor(rows, dtype=torch.long)

    def _cabinets_to_tensor(
        self,
        cabinets: list[ICabinet],
    ) -> torch.LongTensor:
        values = [cabinet.token_id for cabinet in cabinets]
        return torch.tensor(values, dtype=torch.long)

    def _build_condition_mask(
        self,
        true_len: int,
        padded_len: int,
    ) -> torch.BoolTensor:
        mask = torch.zeros(padded_len, dtype=torch.bool)
        mask[:true_len] = True
        return mask

    def _build_cabinet_mask(
        self,
        true_len: int,
        padded_len: int,
    ) -> torch.BoolTensor:
        mask = torch.zeros(padded_len, dtype=torch.bool)
        mask[:true_len] = True
        return mask

    def _build_interleaved_padding_mask(
        self,
        condition_mask: torch.BoolTensor,
        cabinet_mask: torch.BoolTensor,
    ) -> torch.BoolTensor:
        if len(condition_mask) != len(cabinet_mask) + 1:
            raise ValueError(
                "condition_mask length must equal cabinet_mask length + 1"
            )

        total_len = len(condition_mask) + len(cabinet_mask)
        out = torch.ones(total_len, dtype=torch.bool)

        # even positions: conditions
        out[0::2] = ~condition_mask
        # odd positions: cabinets
        out[1::2] = ~cabinet_mask

        return out

    def get_condition_feature_vocab_sizes(self) -> list[int]:
        """
        Returns per-feature vocabulary sizes for the model.

        If encode_condition_pad_for_model=True:
            size = n_states + 1  # extra id for PAD
        else:
            size = n_states
        """
        if self._encode_condition_pad_for_model:
            return [spec.n_states + 1 for spec in self._condition_schema.specs]
        return [spec.n_states for spec in self._condition_schema.specs]

    @property
    def max_conditions_len(self) -> int:
        return self._max_conditions_len

    @property
    def max_cabinets_len(self) -> int:
        return self._max_cabinets_len

    @property
    def condition_schema(self) -> PatientConditionSchema:
        return self._condition_schema

    def summary(self) -> dict[str, Any]:
        return {
            "num_sequences": len(self._records),
            "condition_feature_sizes_raw": [
                spec.n_states for spec in self._condition_schema.specs
            ],
            "condition_feature_sizes_for_model": self.get_condition_feature_vocab_sizes(),
            "max_conditions_per_sequence": self._max_conditions_len,
            "max_cabinets_per_sequence": self._max_cabinets_len,
            "pad_cabinet_token_id": self._pad_cabinet_token_id,
            "encode_condition_pad_for_model": self._encode_condition_pad_for_model,
        }