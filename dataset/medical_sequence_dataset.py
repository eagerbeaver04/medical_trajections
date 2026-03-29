from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from torch.utils.data import Dataset

from structures.medical_sequence import MedicalSequence
from structures.patient_statuses import PatientConditionSchema, PatientCondition
from structures.cabinets import ICabinet
from .sequence_generator import SequenceGenerator


@dataclass
class MedicalSequenceRecord:
    sequence_id: str
    sequence: MedicalSequence
    terminal_status: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)


from typing import Any

from torch.utils.data import Dataset

from structures.cabinets import ICabinet
from structures.medical_sequence import MedicalSequence
from structures.patient_statuses import PatientCondition, PatientConditionSchema

from .medical_sequence_record import MedicalSequenceRecord


class MedicalSequenceDataset(Dataset):
    def __init__(
        self,
        records: list[MedicalSequenceRecord],
        condition_schema: PatientConditionSchema,
        pad_cabinet_token_id: int = 0,
    ) -> None:
        self._records = records
        self._condition_schema = condition_schema
        self._pad_cabinet_token_id = pad_cabinet_token_id
        self._validate()

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> MedicalSequenceRecord:
        return self._records[idx]

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
            raise ValueError(
                f"{sequence_id}: invalid condition length at step {t}"
            )

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
            raise ValueError(
                f"{sequence_id}: invalid cabinet token_id at step {t}"
            )

    def summary(self) -> dict[str, Any]:
        return {
            "num_sequences": len(self._records),
            "condition_feature_sizes": [
                spec.n_states for spec in self._condition_schema.specs
            ],
            "max_conditions_per_sequence": max(
                (len(r.sequence.conditions_sequence) for r in self._records),
                default=0,
            ),
            "max_cabinets_per_sequence": max(
                (len(r.sequence.cabinet_sequence) for r in self._records),
                default=0,
            ),
        }