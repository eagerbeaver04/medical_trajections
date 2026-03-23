from typing import Optional, Callable
from structures.cabinets import ICabinet
from patient_statuses import PatientCondition, get_padding_conditions
import numpy as np
import numpy.typing as npt

class MedicalSequence:
    def __init__(self):
        self._cabinet_sequence: list[ICabinet] = []
        self._conditions_sequence: list[PatientCondition] = []

    def append_cabinet(self, cabinet: ICabinet):
        self._cabinet_sequence.append(cabinet)

    def append_condition(self, condition: PatientCondition):
        self._conditions_sequence.append(condition)

    def get_cabinet_array(self) -> npt.NDArray:
        return np.array(self._cabinet_sequence, dtype=ICabinet)
    
    def get_condition_array(self) -> npt.NDArray:
        return np.array(self._conditions_sequence, dtype=PatientCondition)