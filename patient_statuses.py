from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class IStatus(ABC):
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
    def to_token_id(event: IStatus):
        return event.token_id

    @staticmethod
    def to_key(event: IStatus):
        return event.key

class StartStatus(IStatus):
    @property
    def key(self):
        return "START"
    @property
    def token_id(self):
        return 1

class DeathStatus(IStatus):
    def __init__(self, num_statuses):
        self.num_statuses = num_statuses
    @property
    def key(self):
        return "DEATH"
    @property
    def token_id(self):
        return self.num_statuses + 2   # N+2

class SurviveStatus(IStatus):
    def __init__(self, num_statuses):
        self.num_statuses = num_statuses
    @property
    def key(self):
        return "SURVIVE"
    @property
    def token_id(self):
        return self.num_statuses + 3   # N+3

class PatientStatus(IStatus):
    def __init__(self, index):
        self.index = index  # 2..N+1
    @property
    def key(self):
        return f"STATUS_{self.index}"
    @property
    def token_id(self):
        return self.index   # 2..N+1
    
class PaddingStatus(IStatus):
    @property
    def key(self):
        return "PAD"
    @property
    def token_id(self):
        return 0 
    
def key_to_status(key: str, num_statuses: Optional[int] = None):
    if key == "START":
        return StartStatus()
    elif key == "DEATH":
        if num_statuses is None:
            raise ValueError("num_statuses required for DEATH status")
        return DeathStatus(num_statuses)
    elif key == "SURVIVE":
        if num_statuses is None:
            raise ValueError("num_statuses required for SURVIVE status")
        return SurviveStatus(num_statuses)
    elif key == "PAD":
        return PaddingStatus()
    elif key.startswith("STATUS_"):
        try:
            index = int(key.split("_")[1])
            return PatientStatus(index)
        except (IndexError, ValueError):
            raise ValueError(f"Invalid status key format: {key}")
    else:
        raise ValueError(f"Unknown status key: {key}")
