import numpy as np
from typing import Optional
from events import Event, EndEvent, StartEvent, CabinetEvent, key_to_event
from patient_statuses import IStatus, StartStatus, PatientStatus, DeathStatus, SurviveStatus, key_to_status
from dataclasses import dataclass

class TransitionModel:
    def __init__(self, num_cabinets: int, num_statuses: int,  rng=None):
        self.num_cabinets = num_cabinets
        self.num_statuses = num_statuses
        self._event_transitions = {}  # source_key -> list of (target_key, prob)
        self._status_transitions = {}  # source_key -> list of (target_key, prob)
        self.rng = rng if rng is not None else np.random.default_rng()

    def add_event_transition(self, source_event: Event, target_event: Event, prob: float):
        src_key = source_event.key
        tgt_key = target_event.key
        if src_key not in self._event_transitions:
            self._event_transitions[src_key] = []
        self._event_transitions[src_key].append((tgt_key, prob))

    def add_status_transition(self, source_status: IStatus, target_status: IStatus, prob: float):
        src_key = source_status.key
        tgt_key = target_status.key
        if src_key not in self._status_transitions:
            self._status_transitions[src_key] = []
        self._status_transitions[src_key].append((tgt_key, prob))


    def build_from_matrices(self, start_status_probs: list, status_matrix: list[list], death_status_probs: list, survival_status_probs: list, start_event_probs: list, event_matrix: list[list]):
        status_length = len(start_status_probs)
        assert status_length == self.num_statuses
        assert status_length == len(death_status_probs)
        assert status_length == len(survival_status_probs)
        assert status_length == len(status_matrix)
        assert status_length == len(status_matrix[0])

        event_length = len(start_event_probs)
        assert event_length == self.num_cabinets
        assert event_length == len(event_matrix)
        assert event_length == len(event_matrix[0])

        cabinets = [CabinetEvent(i+2) for i in range(self.num_cabinets)]
        start_event = StartEvent()
        death_event = EndEvent(self.num_cabinets)

        statuses = [PatientStatus(i+2) for i in range(self.num_statuses)]
        start_status = StartStatus()
        death_status = DeathStatus(self.num_statuses)
        survival_status = SurviveStatus(self.num_statuses)

        # Start -> cabinets
        for i, prob in enumerate(start_event_probs):
            if prob > 0:
                self.add_event_transition(start_event, cabinets[i], prob)

        # Start -> statuses
        for i, prob in enumerate(start_status_probs):
            if prob > 0:
                self.add_status_transition(start_status, statuses[i], prob)

        # Cabinet transitions
        for i in range(self.num_cabinets):
            from_cab = cabinets[i]
            # to other cabinets
            for j in range(self.num_cabinets):
                prob = event_matrix[i][j]
                if prob > 0:
                    self.add_event_transition(from_cab, cabinets[j], prob)

        # Status transitions
        for i in range(self.num_statuses):
            from_status = statuses[i]
            # to other cabinets
            for j in range(self.num_statuses):
                prob = status_matrix[i][j]
                if prob > 0:
                    self.add_status_transition(from_status, statuses[j], prob)
            # to Death
            if death_status_probs[i] > 0:
                self.add_status_transition(from_status, death_status, death_status_probs[i])
            # to Survive
            if survival_status_probs[i] > 0:
                self.add_status_transition(from_status, survival_status, survival_status_probs[i])

        self._validate(self._event_transitions)
        self._validate(self._status_transitions)

    def _validate(self, transaction: dict):
        for src_key, targets in transaction.items():
            total = sum(p for _, p in targets)
            if not np.isclose(total, 1.0):
                raise ValueError(f"Probabilities from {src_key} sum to {total}")
    
    @dataclass        
    class Group:
        def __init__(self, status: IStatus, event: Event) -> None:
            self.status = status
            self.event = event

    def next(self, current: Group) -> Optional[Group]:
        src_status_key = current.status.key
        src_event_key = current.event.key
        if src_status_key not in self._status_transitions and src_event_key not in self._event_transitions:
            return None
        elif src_status_key not in self._status_transitions:
            return None
        elif src_event_key not in self._event_transitions:
            raise ValueError("Smth went wrong")
        
        status_targets, status_probs = zip(*self._status_transitions[src_status_key])
        event_targets, event_probs = zip(*self._event_transitions[src_event_key])
        chosen_status_key = self.rng.choice(status_targets, p=status_probs)
        chosen_event_key = self.rng.choice(event_targets, p=event_probs)
        # print(f"From {src_key} chose {chosen_key}")
        return self.Group(status=key_to_status(chosen_status_key, self.num_statuses), event=key_to_event(chosen_event_key, self.num_cabinets))
    
    def is_stopped_status(self, current: Group) -> bool:
        return current.status.key == SurviveStatus(self.num_statuses).key or current.status.key == DeathStatus(self.num_statuses).key