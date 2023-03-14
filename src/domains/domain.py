# Copyright (C) 2021-2022, Ken Tjhia
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Optional, TYPE_CHECKING

import torch as to

if TYPE_CHECKING:
    from search.utils import SearchNode, Trajectory


class Problem:
    def __init__(self, id: str, domain: Domain):
        self.id = id
        self.domain = domain

    def __hash__(self):
        return self.id.__hash__()

    def __eq__(self, other):
        return self.id == other.id


class State(ABC):
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
    def backward_domain(self) -> Domain:
        pass

    @abstractmethod
    def state_tensor(self, state: State) -> to.Tensor:
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
        self, node: SearchNode, other_problem: Domain, num_expanded: int
    ) -> Optional[tuple[Trajectory, Trajectory]]:
        pass
