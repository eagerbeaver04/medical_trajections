from __future__ import annotations
from typing import Optional
import numpy as np
import numpy.typing as npt

class Condition:
    PATIENT_CONDITION_LEN = 10
    def __init__(self, number: int, time: int, name: Optional[str] = None) -> None:
        self._number: int = number
        self._time = time
        self._name: Optional[str] = name

    def __str__(self) -> str:
        if self._name is None: 
            return str(self._number)
        return f"{self._name}_{self._number}"
    
class Conditions:
    def __init__(self, number: int, number_of_conditions: int, time: int, name: Optional[str] = None) -> None:
        self._number = number
        self._name = name
        self._time = time
        self._conditions = np.array(
            [Condition(i, self._time, self._name) for i in range(number_of_conditions)],
            dtype=Condition
        )

    def __str__(self) -> str:
        if self._name is None: 
            return str(self._number)
        return f"{self._name}_{self._number}"

class PatientCondition:
    def __init__(self, conditions: npt.NDArray) -> None:
        assert len(conditions) == Condition.PATIENT_CONDITION_LEN
        self._conditions: npt.NDArray = conditions

def is_patient_dead(patient: PatientCondition) -> bool:
    raise NotImplementedError

def is_patient_survival(patient: PatientCondition) -> bool:
    raise NotImplementedError

def get_padding_conditions() -> PatientCondition:
    raise NotImplementedError