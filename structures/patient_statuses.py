from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ConditionSpec:
    n_states: int


class PatientConditionSchema:
    PAD = -1

    def __init__(self, specs: list[ConditionSpec]) -> None:
        if len(specs) == 0:
            raise ValueError("specs must not be empty")

        for idx, spec in enumerate(specs):
            if spec.n_states <= 0:
                raise ValueError(
                    f"Condition at index {idx} must have positive number of states, "
                    f"got {spec.n_states}"
                )

        self._specs = tuple(specs)

    @property
    def specs(self) -> tuple[ConditionSpec, ...]:
        return self._specs

    def patient_condition_len(self) -> int:
        return len(self._specs)

    def state_count(self, idx: int) -> int:
        return self._specs[idx].n_states

    def validate_number(self, idx: int, number: int, allow_pad: bool = True) -> None:
        if allow_pad and number == self.PAD:
            return

        max_states = self.state_count(idx)
        if not 0 <= number < max_states:
            raise ValueError(
                f"Invalid state {number} for condition index {idx}. "
                f"Allowed values: 0..{max_states - 1}"
            )


@dataclass(frozen=True)
class Condition:
    number: int
    time: int


class PatientCondition:
    def __init__(
        self,
        schema: PatientConditionSchema,
        conditions: npt.NDArray[np.object_],
    ) -> None:
        if len(conditions) != schema.patient_condition_len():
            raise ValueError(
                f"PatientCondition must contain {schema.patient_condition_len()} conditions, "
                f"got {len(conditions)}"
            )

        validated_conditions: list[Condition] = []

        for idx, cond in enumerate(conditions):
            if not isinstance(cond, Condition):
                raise TypeError(
                    f"Element at index {idx} must be Condition, got {type(cond).__name__}"
                )

            schema.validate_number(idx, cond.number, allow_pad=True)
            validated_conditions.append(cond)

        self._schema = schema
        self._conditions = np.array(validated_conditions, dtype=object)

    @property
    def schema(self) -> PatientConditionSchema:
        return self._schema

    @property
    def conditions(self) -> npt.NDArray[np.object_]:
        return self._conditions

    def __getitem__(self, idx: int) -> Condition:
        return self._conditions[idx]

    def __len__(self) -> int:
        return len(self._conditions)

    def as_numbers(self) -> list[int]:
        return [condition.number for condition in self._conditions]

    def is_padding(self) -> bool:
        return all(x == self._schema.PAD for x in self.as_numbers())

    def __repr__(self) -> str:
        return f"PatientCondition({self.as_numbers()})"


def make_patient_condition(
    schema: PatientConditionSchema,
    values: list[int],
    time: int,
) -> PatientCondition:
    if len(values) != schema.patient_condition_len():
        raise ValueError(
            f"values must contain {schema.patient_condition_len()} items, got {len(values)}"
        )

    return PatientCondition(
        schema=schema,
        conditions=np.array(
            [Condition(values[i], time) for i in range(len(values))],
            dtype=object,
        ),
    )


def get_padding_conditions(
    schema: PatientConditionSchema,
    time: int = 0,
) -> PatientCondition:
    return PatientCondition(
        schema=schema,
        conditions=np.array(
            [Condition(schema.PAD, time) for _ in range(schema.patient_condition_len())],
            dtype=object,
        ),
    )