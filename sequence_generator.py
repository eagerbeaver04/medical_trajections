from typing import Optional, Callable
from transisition_model import TransitionModel
from events import StartEvent, Event, PaddingEvent
from patient_statuses import IStatus, StartStatus, PaddingStatus

class MedicalSequence:
    def __init__(self, status_sequence:  Optional[list[IStatus]] = None, event_sequence: Optional[list[Event]] = None):
        if status_sequence is not None:
            self._status_sequence = status_sequence
        else:
            self._status_sequence = []
        if event_sequence is not None:
            self._event_sequence = event_sequence
        else:
            self._event_sequence = []

    def _to_list(self, status_func: Callable[[IStatus], str | int], event_func: Callable[[Event], str | int]) -> list:
        return list(
            zip(list(map(status_func, self._status_sequence)),
                list(map(event_func, self._event_sequence)))
        )

    def to_tokens(self) -> list[tuple[int, int]]:
        return self._to_list(IStatus.to_token_id, Event.to_token_id)

    def print_by_tokens(self):
        print(self.to_tokens())

    def to_keys(self) -> list[tuple[str, str]]:
        return self._to_list(IStatus.to_key, Event.to_key)

    def print_by_keys(self):
        print(self.to_keys())

    def append(self, status: IStatus, event: Event):
        self._status_sequence.append(status)
        self._event_sequence.append(event)

    def __len__(self):
        assert len(self._event_sequence) == len(self._status_sequence)
        return len(self._status_sequence)
    
    def pad(self, length: int):
        assert length >= len(self)
        padded_event_sequence = self._event_sequence + [PaddingEvent()] * (length - len(self._event_sequence))
        padded_status_sequence = self._status_sequence + [PaddingStatus()] * (length - len(self._status_sequence))
        self._event_sequence = padded_event_sequence
        self._status_sequence = padded_status_sequence


class MedicalSequenceGenerator:
    def __init__(self, model: TransitionModel):
        self.model = model
        self.num_cabinets = model.num_cabinets

    def generate_sequence(self) -> MedicalSequence:
        sequence = MedicalSequence()
        current_event = StartEvent()
        current_status = StartStatus()
        current = TransitionModel.Group(current_status, current_event)
        while not self.model.is_stopped_status(current):
            print(current.status.key, current.event.key)
            sequence.append(current.status, current.event)
            new_current = self.model.next(current)
            if new_current is None:
                raise ValueError(f"Incorrect current = none for {current.status.key}")
            current = new_current
        sequence.append(current.status, current.event)
        return sequence

    def generate_many(self, n) -> list[MedicalSequence]:
        return [self.generate_sequence() for _ in range(n)]