from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch as to

if TYPE_CHECKING:
    from search.utils import SearchNode, Trajectory


class State(ABC):
    # @abstractmethod
    # def __repr__(self):
    #     pass

    @abstractmethod
    def __eq__(self) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
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
    def reset(self) -> State:
        pass

    @abstractmethod
    def backward_problem(self) -> Domain:
        pass

    @abstractmethod
    def state_tensor(
        self, state: State, device: to.device = to.device("cpu")
    ) -> to.Tensor:
        pass

    @abstractmethod
    def update(self, node: SearchNode):
        pass

    @abstractmethod
    def actions(self, action, state: State) -> list:
        pass

    @abstractmethod
    def actions_unpruned(self, state: State) -> list:
        pass

    @abstractmethod
    def result(self, state: State, action) -> State:
        pass

    @abstractmethod
    def try_make_solution(
        self, state: State, other_problem: Domain, num_expanded: int, device: to.device
    ) -> Optional[tuple[Trajectory, Trajectory]]:
        pass
