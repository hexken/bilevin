from abc import ABC, abstractmethod


class Agent(ABC):
    @property
    @classmethod
    @abstractmethod
    def bidirectional(cls):
        pass

    @abstractmethod
    def search(self, problem):
        pass
