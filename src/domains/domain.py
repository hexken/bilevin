from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import torch as to

from search.utils import SearchNode, Trajectory


class State(ABC):
    # @abstractmethod
    # def __repr__(self):
    #     pass

    @abstractmethod
    def __eq__(self):
        pass

    @abstractmethod
    def __hash__(self):
        pass


class Domain(ABC):
    @property
    @abstractmethod
    def num_actions(self):
        pass

    @property
    @abstractmethod
    def in_channels(self):
        pass

    @property
    @abstractmethod
    def state_width(self):
        pass

    @abstractmethod
    def is_goal(self, state: State) -> bool:
        pass

    @abstractmethod
    def reset(self, state: State) -> bool:
        pass

    @abstractmethod
    def state_tensor(self):
        pass

    @abstractmethod
    def update(self, node: SearchNode):
        pass

    @abstractmethod
    def actions(self, state: State):
        pass

    @abstractmethod
    def actions_unpruned(self, state: State):
        pass

    @abstractmethod
    def result(self, action, state: State):
        pass

    @abstractmethod
    def try_make_solution(
        self, state: State, other_problem: Domain, num_expanded: int, device: to.device
    ) -> Optional[tuple[Trajectory, Trajectory]]:
        pass
