from abc import ABC, abstractmethod
from typing import Optional


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
    def reset(self, state: State) -> bool:
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


class Problem:
    def __init__(
        self,
        name: str,
        domain: Domain,
    ):
        self.name = name
        self.domain = domain
