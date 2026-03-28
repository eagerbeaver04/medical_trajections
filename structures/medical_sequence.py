import numpy as np
import numpy.typing as npt

from structures.cabinets import ICabinet
from structures.patient_statuses import PatientCondition


class MedicalSequence:
    def __init__(self) -> None:
        self._cabinet_sequence: list[ICabinet] = []
        self._conditions_sequence: list[PatientCondition] = []

    def append_cabinet(self, cabinet: ICabinet) -> None:
        self._cabinet_sequence.append(cabinet)

    def append_condition(self, condition: PatientCondition) -> None:
        self._conditions_sequence.append(condition)

    @property
    def cabinet_sequence(self) -> list[ICabinet]:
        return self._cabinet_sequence

    @property
    def conditions_sequence(self) -> list[PatientCondition]:
        return self._conditions_sequence

    def get_cabinet_array(self) -> npt.NDArray[np.object_]:
        return np.array(self._cabinet_sequence, dtype=object)

    def get_condition_array(self) -> npt.NDArray[np.object_]:
        return np.array(self._conditions_sequence, dtype=object)

    def __len__(self) -> int:
        return len(self._conditions_sequence)

    def __repr__(self) -> str:
        return (
            f"MedicalSequence("
            f"n_conditions={len(self._conditions_sequence)}, "
            f"n_cabinets={len(self._cabinet_sequence)}"
            f")"
        )