from abc import ABC, abstractmethod
from typing import Optional


from enum import IntEnum


class DirAction(IntEnum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3


class State(ABC):
    @abstractmethod
    def as_tensor(self):
        pass


class Domain(ABC):
    @property
    @classmethod
    @abstractmethod
    def num_actions(cls):
        pass

    @abstractmethod
    def is_goal(self, state: State) -> bool:
        pass

    @abstractmethod
    def actions(self, state: State):
        pass

    @abstractmethod
    def actions_unpruned(self, state: State):
        pass

    @abstractmethod
    def result(self, action: int, state: State):
        pass

    @abstractmethod
    def state_equal(self, state1, state2):
        pass


class Problem:
    def __init__(
        self, domain: Domain, initial_state: State, name: Optional[str] = None
    ):
        self.domain = domain
        self.initial_state = initial_state
        self.name = name
