import numpy as np
from typing import Any

from structures.patient_statuses import PatientCondition, PatientConditionSchema, ConditionSpec, make_patient_condition
from structures.medical_sequence import MedicalSequence
from .condition_cabinet_relation import Relation, PatientTerminalStatus

class SequenceGenerator():
    def __init__(self) -> None:

        schema = PatientConditionSchema(
        specs=[
            ConditionSpec(n_states=3),
            ConditionSpec(n_states=2),
            ConditionSpec(n_states=4),
        ]
        )

        # 2 кабинета: token_id будут 2 и 3
        relation = Relation(
            schema=schema,
            n_cabinets=2,
            rng_seed=42,
        )

        # Априорная вероятность кабинетов
        relation.set_cabinet_prior(np.array([0.6, 0.4]))

        # Вероятность кабинета по наблюдаемому состоянию condition/state
        # condition 0
        relation.set_cabinet_condition_likelihood(0, 0, np.array([0.8, 0.2]))
        relation.set_cabinet_condition_likelihood(0, 1, np.array([0.5, 0.5]))
        relation.set_cabinet_condition_likelihood(0, 2, np.array([0.2, 0.8]))

        # condition 1
        relation.set_cabinet_condition_likelihood(1, 0, np.array([0.7, 0.3]))
        relation.set_cabinet_condition_likelihood(1, 1, np.array([0.3, 0.7]))

        # condition 2
        relation.set_cabinet_condition_likelihood(2, 0, np.array([0.9, 0.1]))
        relation.set_cabinet_condition_likelihood(2, 1, np.array([0.6, 0.4]))
        relation.set_cabinet_condition_likelihood(2, 2, np.array([0.4, 0.6]))
        relation.set_cabinet_condition_likelihood(2, 3, np.array([0.1, 0.9]))

        # Переходы состояний под действием кабинета 0 (token_id=2)
        # condition 0 has 3 states
        relation.set_condition_transition(0, 0, 0, np.array([0.7, 0.2, 0.1]))
        relation.set_condition_transition(0, 0, 1, np.array([0.3, 0.5, 0.2]))
        relation.set_condition_transition(0, 0, 2, np.array([0.2, 0.4, 0.4]))

        # condition 1 has 2 states
        relation.set_condition_transition(0, 1, 0, np.array([0.8, 0.2]))
        relation.set_condition_transition(0, 1, 1, np.array([0.4, 0.6]))

        # condition 2 has 4 states
        relation.set_condition_transition(0, 2, 0, np.array([0.6, 0.3, 0.1, 0.0]))
        relation.set_condition_transition(0, 2, 1, np.array([0.2, 0.5, 0.2, 0.1]))
        relation.set_condition_transition(0, 2, 2, np.array([0.1, 0.3, 0.4, 0.2]))
        relation.set_condition_transition(0, 2, 3, np.array([0.0, 0.2, 0.3, 0.5]))

        # Переходы состояний под действием кабинета 1 (token_id=3)
        relation.set_condition_transition(1, 0, 0, np.array([0.4, 0.4, 0.2]))
        relation.set_condition_transition(1, 0, 1, np.array([0.2, 0.5, 0.3]))
        relation.set_condition_transition(1, 0, 2, np.array([0.1, 0.3, 0.6]))

        relation.set_condition_transition(1, 1, 0, np.array([0.5, 0.5]))
        relation.set_condition_transition(1, 1, 1, np.array([0.2, 0.8]))

        relation.set_condition_transition(1, 2, 0, np.array([0.3, 0.3, 0.3, 0.1]))
        relation.set_condition_transition(1, 2, 1, np.array([0.1, 0.4, 0.3, 0.2]))
        relation.set_condition_transition(1, 2, 2, np.array([0.0, 0.2, 0.5, 0.3]))
        relation.set_condition_transition(1, 2, 3, np.array([0.0, 0.1, 0.3, 0.6]))

        # Веса терминальных исходов
        # Чем больше weight для состояния, тем сильнее вклад
        # condition 0
        relation.set_death_likelihood(0, np.array([0.10, 0.10, 0.20]))
        relation.set_survival_likelihood(0, np.array([0.20, 0.10, 0.12]))

        relation.set_death_likelihood(1, np.array([0.12, 0.15]))
        relation.set_survival_likelihood(1, np.array([0.15, 0.05]))

        relation.set_death_likelihood(2, np.array([0.10, 0.12, 0.12, 0.20]))
        relation.set_survival_likelihood(2, np.array([0.20, 0.12, 0.08, 0.10]))

        self._relation: Relation = relation
        self._schema: PatientConditionSchema = schema

    def _generate_initial_patient_condition(self) -> PatientCondition:
        return make_patient_condition(
        schema=self._schema,
        values=[1, 0, 2],
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
