from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class ICabinet(ABC):
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
    def to_token_id(event: ICabinet):
        return event.token_id

    @staticmethod
    def to_key(event: ICabinet):
        return event.key

class Cabinet(ICabinet):
    def __init__(self, time: int, index: int):
        self._time = time
        self._index = index  # 2..N+1
    @property
    def key(self):
        return f"CABINET_{self._index}"
    @property
    def token_id(self):
        return self._index   # 2..N+1
    
class PaddingCabinet(ICabinet):
    @property
    def key(self):
        return "PAD"
    @property
    def token_id(self):
        return 0 