from abc import ABC, abstractmethod
from typing import Generic, Optional, TYPE_CHECKING, TypeVar


class State(ABC):
    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass


TState = TypeVar("TState", bound=State)
