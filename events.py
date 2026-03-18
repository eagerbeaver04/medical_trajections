from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class Event(ABC):
    @property
    @abstractmethod
    def key(self) -> str:
        """Unique identifier for transition logic."""
        pass

    @property
    @abstractmethod
    def token_id(self) -> int:
        """Integer token for ML model."""
        pass

    @staticmethod
    def to_token_id(event: Event):
        return event.token_id

    @staticmethod
    def to_key(event: Event):
        return event.key
    

class StartEvent(Event):
    @property
    def key(self):
        return "START"
    @property
    def token_id(self):
        return 1
    

class EndEvent(Event):
    def __init__(self, num_cabinets):
        self.num_cabinets = num_cabinets
    @property
    def key(self):
        return "DEATH"
    @property
    def token_id(self):
        return self.num_cabinets + 2   # N+2

class CabinetEvent(Event):
    def __init__(self, index):
        self.index = index  # 2..N+1
    @property
    def key(self):
        return f"CABINET_{self.index}"
    @property
    def token_id(self):
        return self.index   # 2..N+1
    
class PaddingEvent(Event):
    @property
    def key(self):
        return "PAD"
    @property
    def token_id(self):
        return 0 
    
def key_to_event(key: str, var: Optional[int] = None) -> Event:
    if key == "START":
        return StartEvent()
    
    elif key == "DEATH":
        if var is None:
            raise ValueError("num_cabinets (var) required for DEATH event")
        return EndEvent(var)
    
    elif key == "PAD":
        return PaddingEvent()
    
    elif key.startswith("CABINET_"):
        try:
            index = int(key.split("_")[1])
            return CabinetEvent(index)
        except (IndexError, ValueError):
            if var is not None and isinstance(var, int):
                return CabinetEvent(var)
            else:
                raise ValueError(f"Invalid cabinet event key format: {key} or missing cabinet index")
    
    else:
        raise ValueError(f"Unknown event key: {key}")