from abc import ABC, abstractmethod


class Environment(ABC):
    @abstractmethod
    def successors(self):
        pass

    @abstractmethod
    def is_solution(self):
        pass

    @abstractmethod
    def apply_action(self, action):
        pass

    @abstractmethod
    def state_tensor(self):
        pass

    def get_image_representation(self):
        pass

    def heuristic_value(self):
        pass

    def reset(self):
        pass

    def copy(self):
        pass

    @abstractmethod
    def state_equal(self, other):
        pass

    @property
    @classmethod
    @abstractmethod
    def num_actions(cls):
        raise NotImplementedError
