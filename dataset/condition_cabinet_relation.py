from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import numpy.typing as npt

from structures.cabinets import ICabinet, Cabinet, PaddingCabinet
from structures.patient_statuses import (
    Condition,
    PatientCondition,
    PatientConditionSchema,
    get_padding_conditions,
)


class PatientTerminalStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    SURVIVED = "survived"
    DEAD = "dead"


class Relation:
    def __init__(
        self,
        schema: PatientConditionSchema,
        n_cabinets: int,
        rng_seed: Optional[int] = None,
        cabinet_prior: Optional[npt.NDArray[np.float64]] = None,
        cabinet_condition_likelihood: Optional[npt.NDArray[np.float64]] = None,
        condition_transition: Optional[npt.NDArray[np.float64]] = None,
        death_likelihood: Optional[npt.NDArray[np.float64]] = None,
        survival_likelihood: Optional[npt.NDArray[np.float64]] = None,
        terminal_prior: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        if n_cabinets <= 0:
            raise ValueError("n_cabinets must be positive")

        self._schema = schema
        self._n_conditions = schema.patient_condition_len()
        self._n_cabinets = n_cabinets
        self._max_states = max(spec.n_states for spec in schema.specs)
        self._rng = np.random.default_rng(rng_seed)

        self._cabinet_prior = (
            self._init_uniform_cabinet_prior()
            if cabinet_prior is None
            else np.array(cabinet_prior, dtype=float)
        )

        self._cabinet_condition_likelihood = (
            self._init_uniform_cabinet_condition_likelihood()
            if cabinet_condition_likelihood is None
            else np.array(cabinet_condition_likelihood, dtype=float)
        )

        self._condition_transition = (
            self._init_identity_like_condition_transition()
            if condition_transition is None
            else np.array(condition_transition, dtype=float)
        )

        self._death_likelihood = (
            self._init_neutral_terminal_likelihood()
            if death_likelihood is None
            else np.array(death_likelihood, dtype=float)
        )

        self._survival_likelihood = (
            self._init_neutral_terminal_likelihood()
            if survival_likelihood is None
            else np.array(survival_likelihood, dtype=float)
        )

        self._terminal_prior = (
            np.array([1.0, 1.0, 1.0], dtype=float)
            if terminal_prior is None
            else np.array(terminal_prior, dtype=float)
        )

        self._validate_shapes()
        self._normalize_all()

    @property
    def schema(self) -> PatientConditionSchema:
        return self._schema

    def _init_uniform_cabinet_prior(self) -> npt.NDArray[np.float64]:
        return np.ones(self._n_cabinets, dtype=float) / self._n_cabinets

    def _init_uniform_cabinet_condition_likelihood(self) -> npt.NDArray[np.float64]:
        arr = np.zeros(
            (self._n_conditions, self._max_states, self._n_cabinets),
            dtype=float,
        )
        for i in range(self._n_conditions):
            n_states = self._schema.state_count(i)
            arr[i, :n_states, :] = 1.0
        return arr

    def _init_identity_like_condition_transition(self) -> npt.NDArray[np.float64]:
        arr = np.zeros(
            (self._n_cabinets, self._n_conditions, self._max_states, self._max_states),
            dtype=float,
        )

        for c in range(self._n_cabinets):
            for i in range(self._n_conditions):
                n_states = self._schema.state_count(i)
                for old_state in range(n_states):
                    arr[c, i, old_state, old_state] = 1.0

        return arr

    def _init_neutral_terminal_likelihood(self) -> npt.NDArray[np.float64]:
        arr = np.zeros((self._n_conditions, self._max_states), dtype=float)
        for i in range(self._n_conditions):
            n_states = self._schema.state_count(i)
            arr[i, :n_states] = 1.0
        return arr

    def _validate_shapes(self) -> None:
        if self._cabinet_prior.shape != (self._n_cabinets,):
            raise ValueError(
                f"cabinet_prior must have shape ({self._n_cabinets},), "
                f"got {self._cabinet_prior.shape}"
            )

        expected_likelihood_shape = (
            self._n_conditions,
            self._max_states,
            self._n_cabinets,
        )
        if self._cabinet_condition_likelihood.shape != expected_likelihood_shape:
            raise ValueError(
                "cabinet_condition_likelihood must have shape "
                f"{expected_likelihood_shape}, got {self._cabinet_condition_likelihood.shape}"
            )

        expected_transition_shape = (
            self._n_cabinets,
            self._n_conditions,
            self._max_states,
            self._max_states,
        )
        if self._condition_transition.shape != expected_transition_shape:
            raise ValueError(
                f"condition_transition must have shape {expected_transition_shape}, "
                f"got {self._condition_transition.shape}"
            )

        expected_terminal_shape = (self._n_conditions, self._max_states)
        if self._death_likelihood.shape != expected_terminal_shape:
            raise ValueError(
                f"death_likelihood must have shape {expected_terminal_shape}, "
                f"got {self._death_likelihood.shape}"
            )

        if self._survival_likelihood.shape != expected_terminal_shape:
            raise ValueError(
                f"survival_likelihood must have shape {expected_terminal_shape}, "
                f"got {self._survival_likelihood.shape}"
            )

        if self._terminal_prior.shape != (3,):
            raise ValueError(
                f"terminal_prior must have shape (3,), got {self._terminal_prior.shape}"
            )

    def _normalize_vector(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        total = float(np.sum(x))
        if total <= 0.0:
            raise ValueError("Cannot normalize vector with non-positive sum")
        return x / total

    def _normalize_all(self) -> None:
        self._cabinet_prior = self._normalize_vector(self._cabinet_prior)
        self._terminal_prior = self._normalize_vector(self._terminal_prior)

        for i in range(self._n_conditions):
            n_states = self._schema.state_count(i)

            for state in range(n_states):
                row = self._cabinet_condition_likelihood[i, state, :]
                self._cabinet_condition_likelihood[i, state, :] = self._normalize_vector(row)

            death_row_sum = float(np.sum(self._death_likelihood[i, :n_states]))
            if death_row_sum <= 0.0:
                raise ValueError(f"death_likelihood for condition {i} must contain positive values")

            survival_row_sum = float(np.sum(self._survival_likelihood[i, :n_states]))
            if survival_row_sum <= 0.0:
                raise ValueError(
                    f"survival_likelihood for condition {i} must contain positive values"
                )

            for c in range(self._n_cabinets):
                for old_state in range(n_states):
                    row = self._condition_transition[c, i, old_state, :n_states]
                    self._condition_transition[c, i, old_state, :n_states] = self._normalize_vector(row)

    def _validate_patient_condition(self, patient_condition: PatientCondition) -> None:
        if patient_condition.schema is not self._schema:
            raise ValueError("PatientCondition schema does not match Relation schema")

        if len(patient_condition) != self._n_conditions:
            raise ValueError(
                f"PatientCondition must contain {self._n_conditions} conditions, "
                f"got {len(patient_condition)}"
            )

        for i, cond in enumerate(patient_condition.conditions):
            self._schema.validate_number(i, cond.number, allow_pad=True)

    def _sample_index(self, probs: npt.NDArray[np.float64]) -> int:
        probs = np.array(probs, dtype=float)
        probs = self._normalize_vector(probs)
        return int(self._rng.choice(len(probs), p=probs))

    def _make_patient_condition_from_numbers(
        self,
        values: list[int],
        time: int,
    ) -> PatientCondition:
        return PatientCondition(
            schema=self._schema,
            conditions=np.array(
                [Condition(values[i], time) for i in range(len(values))],
                dtype=object,
            ),
        )

    def set_cabinet_prior(self, cabinet_prior: npt.NDArray[np.float64]) -> None:
        cabinet_prior = np.array(cabinet_prior, dtype=float)
        if cabinet_prior.shape != (self._n_cabinets,):
            raise ValueError(
                f"cabinet_prior must have shape ({self._n_cabinets},), got {cabinet_prior.shape}"
            )
        self._cabinet_prior = self._normalize_vector(cabinet_prior)

    def set_cabinet_condition_likelihood(
        self,
        condition_idx: int,
        state: int,
        weights_for_cabinets: npt.NDArray[np.float64],
    ) -> None:
        self._schema.validate_number(condition_idx, state, allow_pad=False)
        weights = np.array(weights_for_cabinets, dtype=float)
        if weights.shape != (self._n_cabinets,):
            raise ValueError(
                f"weights_for_cabinets must have shape ({self._n_cabinets},), "
                f"got {weights.shape}"
            )
        self._cabinet_condition_likelihood[condition_idx, state, :] = self._normalize_vector(
            weights
        )

    def set_condition_transition(
        self,
        cabinet_idx: int,
        condition_idx: int,
        old_state: int,
        probs_to_new_states: npt.NDArray[np.float64],
    ) -> None:
        if not 0 <= cabinet_idx < self._n_cabinets:
            raise ValueError(
                f"cabinet_idx must be in [0, {self._n_cabinets - 1}], got {cabinet_idx}"
            )

        self._schema.validate_number(condition_idx, old_state, allow_pad=False)
        n_states = self._schema.state_count(condition_idx)

        probs = np.array(probs_to_new_states, dtype=float)
        if probs.shape != (n_states,):
            raise ValueError(
                f"probs_to_new_states must have shape ({n_states},), got {probs.shape}"
            )

        self._condition_transition[cabinet_idx, condition_idx, old_state, :] = 0.0
        self._condition_transition[cabinet_idx, condition_idx, old_state, :n_states] = (
            self._normalize_vector(probs)
        )

    def set_death_likelihood(
        self,
        condition_idx: int,
        weights_for_states: npt.NDArray[np.float64],
    ) -> None:
        n_states = self._schema.state_count(condition_idx)
        weights = np.array(weights_for_states, dtype=float)

        if weights.shape != (n_states,):
            raise ValueError(
                f"weights_for_states must have shape ({n_states},), got {weights.shape}"
            )

        self._death_likelihood[condition_idx, :] = 0.0
        self._death_likelihood[condition_idx, :n_states] = self._normalize_vector(
            weights
        )

    def set_survival_likelihood(
        self,
        condition_idx: int,
        weights_for_states: npt.NDArray[np.float64],
    ) -> None:
        n_states = self._schema.state_count(condition_idx)
        weights = np.array(weights_for_states, dtype=float)

        if weights.shape != (n_states,):
            raise ValueError(
                f"weights_for_states must have shape ({n_states},), got {weights.shape}"
            )

        self._survival_likelihood[condition_idx, :] = 0.0
        self._survival_likelihood[condition_idx, :n_states] = self._normalize_vector(
            weights
        )

    def terminal_status_distribution(
        self,
        patient_condition: PatientCondition,
    ) -> dict[PatientTerminalStatus, float]:
        self._validate_patient_condition(patient_condition)

        if patient_condition.is_padding():
            return {
                PatientTerminalStatus.IN_PROGRESS: 1.0,
                PatientTerminalStatus.SURVIVED: 0.0,
                PatientTerminalStatus.DEAD: 0.0,
            }

        dead_score = float(self._terminal_prior[2])
        survived_score = float(self._terminal_prior[1])
        in_progress_score = float(self._terminal_prior[0])

        for i, cond in enumerate(patient_condition.conditions):
            state = cond.number
            if state == self._schema.PAD:
                continue

            dead_score *= float(self._death_likelihood[i, state])
            survived_score *= float(self._survival_likelihood[i, state])

        probs = np.array(
            [
                in_progress_score,
                survived_score,
                dead_score,
            ],
            dtype=float,
        )
        probs = self._normalize_vector(probs)

        return {
            PatientTerminalStatus.IN_PROGRESS: float(probs[0]),
            PatientTerminalStatus.SURVIVED: float(probs[1]),
            PatientTerminalStatus.DEAD: float(probs[2]),
        }

    def determine_terminal_status(
        self,
        patient_condition: PatientCondition,
    ) -> PatientTerminalStatus:
        distribution = self.terminal_status_distribution(patient_condition)
        probs = np.array(
            [
                distribution[PatientTerminalStatus.IN_PROGRESS],
                distribution[PatientTerminalStatus.SURVIVED],
                distribution[PatientTerminalStatus.DEAD],
            ],
            dtype=float,
        )
        idx = self._sample_index(probs)

        if idx == 0:
            return PatientTerminalStatus.IN_PROGRESS
        if idx == 1:
            return PatientTerminalStatus.SURVIVED
        return PatientTerminalStatus.DEAD

    # def determine_terminal_status_argmax(
    #     self,
    #     patient_condition: PatientCondition,
    # ) -> PatientTerminalStatus:
    #     distribution = self.terminal_status_distribution(patient_condition)
    #     probs = np.array(
    #         [
    #             distribution[PatientTerminalStatus.IN_PROGRESS],
    #             distribution[PatientTerminalStatus.SURVIVED],
    #             distribution[PatientTerminalStatus.DEAD],
    #         ],
    #         dtype=float,
    #     )
    #     idx = int(np.argmax(probs))

    #     if idx == 0:
    #         return PatientTerminalStatus.IN_PROGRESS
    #     if idx == 1:
    #         return PatientTerminalStatus.SURVIVED
    #     return PatientTerminalStatus.DEAD

    def from_condition_to_cabinet(self, patient_condition: PatientCondition) -> ICabinet:
        self._validate_patient_condition(patient_condition)

        if patient_condition.is_padding():
            return PaddingCabinet()

        scores = np.array(self._cabinet_prior, dtype=float)

        for i, cond in enumerate(patient_condition.conditions):
            state = cond.number
            if state == self._schema.PAD:
                continue
            scores *= self._cabinet_condition_likelihood[i, state, :]

        if float(np.sum(scores)) <= 0.0:
            scores = np.ones(self._n_cabinets, dtype=float) / self._n_cabinets
        else:
            scores = self._normalize_vector(scores)

        cabinet_zero_based = self._sample_index(scores)
        cabinet_index = cabinet_zero_based + 2
        return Cabinet(time=patient_condition[0].time, index=cabinet_index)

    def from_cabinet_to_condition(
        self,
        current_condition: PatientCondition,
        cabinet: ICabinet,
        next_time: Optional[int] = None,
    ) -> PatientCondition:
        self._validate_patient_condition(current_condition)

        if current_condition.is_padding():
            return get_padding_conditions(
                schema=self._schema,
                time=0 if next_time is None else next_time,
            )

        if isinstance(cabinet, PaddingCabinet) or cabinet.token_id == 0:
            return get_padding_conditions(
                schema=self._schema,
                time=current_condition[0].time if next_time is None else next_time,
            )

        cabinet_zero_based = cabinet.token_id - 2
        if not 0 <= cabinet_zero_based < self._n_cabinets:
            raise ValueError(
                f"Cabinet token_id {cabinet.token_id} is out of supported range"
            )

        time_value = current_condition[0].time if next_time is None else next_time
        next_values: list[int] = []

        for i, cond in enumerate(current_condition.conditions):
            old_state = cond.number

            if old_state == self._schema.PAD:
                next_values.append(self._schema.PAD)
                continue

            n_states = self._schema.state_count(i)
            probs = self._condition_transition[
                cabinet_zero_based,
                i,
                old_state,
                :n_states,
            ]
            new_state = self._sample_index(probs)
            next_values.append(new_state)

        return self._make_patient_condition_from_numbers(next_values, time_value)