import numpy as np
from typing import Optional

from structures.patient_statuses import (
    PatientCondition,
    PatientConditionSchema,
    ConditionSpec,
    make_patient_condition,
)
from structures.medical_sequence import MedicalSequence
from .condition_cabinet_relation import Relation, PatientTerminalStatus


class SequenceGenerator:
    def __init__(
        self,
        rng_seed: Optional[int] = None,
        n_cabinets: int = 6,
        condition_specs: Optional[list[int]] = None,
        transition_bias: float = 3.0,
        self_loop_bias: float = 2.0,
    ) -> None:
        """
        Args:
            rng_seed: seed RNG для воспроизводимости.
            n_cabinets: количество кабинетов (>=1).
            condition_specs: количество состояний по каждому condition, например [3, 2, 4].
            transition_bias: насколько сильно каждый кабинет "тянет" к своему целевому состоянию.
            self_loop_bias: бонус к сохранению текущего состояния.
        """
        if n_cabinets <= 0:
            raise ValueError("n_cabinets must be positive")

        if condition_specs is None:
            condition_specs = [3, 2, 4]

        if any(n <= 1 for n in condition_specs):
            raise ValueError("Each condition must have at least 2 states")

        self._rng = np.random.default_rng(rng_seed)

        schema = PatientConditionSchema(
            specs=[ConditionSpec(n_states=n) for n in condition_specs]
        )

        relation = Relation(
            schema=schema,
            n_cabinets=n_cabinets,
            rng_seed=rng_seed,
        )

        self._init_cabinet_prior(relation, n_cabinets)
        self._init_cabinet_condition_likelihood(relation, schema, n_cabinets)
        self._init_condition_transitions(
            relation=relation,
            schema=schema,
            n_cabinets=n_cabinets,
            transition_bias=transition_bias,
            self_loop_bias=self_loop_bias,
        )
        self._init_terminal_likelihoods(relation, schema)

        self._relation = relation
        self._schema = schema

    def _init_cabinet_prior(self, relation: Relation, n_cabinets: int) -> None:
        # Случайный prior по кабинетам
        prior = self._rng.dirichlet(np.ones(n_cabinets))
        relation.set_cabinet_prior(prior)

    def _init_cabinet_condition_likelihood(
        self,
        relation: Relation,
        schema: PatientConditionSchema,
        n_cabinets: int,
    ) -> None:
        # Для каждого condition/state строим распределение по кабинетам.
        # Делается с "предпочтительным" кабинетом, чтобы кабинеты отличались.
        for condition_idx, spec in enumerate(schema.specs):
            n_states = spec.n_states

            for state in range(n_states):
                alpha = np.ones(n_cabinets, dtype=float)

                if n_cabinets == 1:
                    preferred_cabinet = 0
                elif n_states == 1:
                    preferred_cabinet = 0
                else:
                    preferred_cabinet = int(
                        round(state * (n_cabinets - 1) / (n_states - 1))
                    )

                alpha[preferred_cabinet] += 3.0
                weights = self._rng.dirichlet(alpha)
                relation.set_cabinet_condition_likelihood(
                    condition_idx=condition_idx,
                    state=state,
                    weights_for_cabinets=weights,
                )

    def _init_condition_transitions(
        self,
        relation: Relation,
        schema: PatientConditionSchema,
        n_cabinets: int,
        transition_bias: float,
        self_loop_bias: float,
    ) -> None:
        # Генерируем P(next_state | cabinet, condition, old_state)
        # с bias к "целевому" состоянию для каждого кабинета + bias к self-loop.
        for cabinet_idx in range(n_cabinets):
            for condition_idx, spec in enumerate(schema.specs):
                n_states = spec.n_states

                if n_cabinets == 1:
                    target_state = n_states // 2
                else:
                    target_state = int(
                        round(cabinet_idx * (n_states - 1) / (n_cabinets - 1))
                    )

                for old_state in range(n_states):
                    alpha = np.ones(n_states, dtype=float)
                    alpha[target_state] += transition_bias
                    alpha[old_state] += self_loop_bias

                    probs = self._rng.dirichlet(alpha)
                    relation.set_condition_transition(
                        cabinet_idx=cabinet_idx,
                        condition_idx=condition_idx,
                        old_state=old_state,
                        probs_to_new_states=probs,
                    )

    def _init_terminal_likelihoods(
        self,
        relation: Relation,
        schema: PatientConditionSchema,
    ) -> None:
        # Автогенерация весов terminal исходов.
        # Логика: для каждого condition состояние "правее" слегка повышает death,
        # "левее" — survival, плюс шум.
        for condition_idx, spec in enumerate(schema.specs):
            n_states = spec.n_states

            # Базовые тренды + шум
            x = np.linspace(0.0, 1.0, n_states)
            death = 0.5 + 1.5 * x + 0.2 * self._rng.random(n_states)
            survival = 0.5 + 1.5 * (1.0 - x) + 0.2 * self._rng.random(n_states)

            relation.set_death_likelihood(condition_idx, death)
            relation.set_survival_likelihood(condition_idx, survival)

    def _generate_initial_patient_condition(self) -> PatientCondition:
        # Старт из "средних" состояний каждого condition
        values = [spec.n_states // 2 for spec in self._schema.specs]
        return make_patient_condition(
            schema=self._schema,
            values=values,
            time=0,
        )

    def generate_sequence(
        self,
        max_steps: int = 50,
    ) -> tuple[MedicalSequence, PatientTerminalStatus]:
        current = self._generate_initial_patient_condition()
        sequence = MedicalSequence()
        sequence.append_condition(current)

        for _ in range(max_steps):
            terminal_status = self._relation.determine_terminal_status(current)

            if terminal_status in (
                PatientTerminalStatus.SURVIVED,
                PatientTerminalStatus.DEAD,
            ):
                return sequence, terminal_status

            cabinet = self._relation.from_condition_to_cabinet(current)
            next_condition = self._relation.from_cabinet_to_condition(
                current_condition=current,
                cabinet=cabinet,
                next_time=current[0].time + 1,
            )

            sequence.append_cabinet(cabinet)
            sequence.append_condition(next_condition)
            current = next_condition

        return sequence, PatientTerminalStatus.IN_PROGRESS

    @property
    def schema(self) -> PatientConditionSchema:
        return self._schema