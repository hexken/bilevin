from abc import ABC, abstractmethod
from domains.domain import Domain


class Agent(ABC):
    @property
    @classmethod
    @abstractmethod
    def bidirectional(cls):
        pass

    @property
    @classmethod
    @abstractmethod
    def trainable(cls):
        pass

    @abstractmethod
    def search(self, problem: tuple[int, Domain]):
        pass
