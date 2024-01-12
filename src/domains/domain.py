from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

import torch as to
from torch import Tensor, full

if TYPE_CHECKING:
    from search.node import SearchNode
    from search.traj import Trajectory, from_common_node
    from search.agent import Agent


class State(ABC):
    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass


class Domain(ABC):
    def __init__(self, forward: bool = True):
        self.aux_closed: dict = {}
        self.forward: bool = forward
        self.initial_state: State
        self.goal_state_t: Optional[Tensor] = None

    @abstractmethod
    def reset(self) -> State:
        pass

    def _reset(self) -> State:
        self.aux_closed = {}
        return self.initial_state

    def update(self, node: SearchNode):
        self.aux_closed[node.state.__hash__()] = node

    def actions(self, parent_action, state: State) -> tuple[list, Tensor]:
        actions = self._actions(parent_action, state)
        mask = full((self.num_actions,), True, dtype=to.bool)
        mask[actions] = False
        return actions, mask

    def actions_unpruned(self, state: State) -> tuple[list, Tensor]:
        actions = self._actions_unpruned(state)
        mask = full((self.num_actions,), True, dtype=to.bool)
        mask[actions] = False
        return actions, mask

    def is_merge_goal(self, state, other_domain) -> Optional[SearchNode]:
        hsh = state.__hash__()
        if hsh in other_domain.aux_closed:  # solution found
            other_node = other_domain.aux_closed[hsh]
            return other_node
        else:
            return None

    def try_make_solution(
        self,
        agent: Agent,
        node: SearchNode,
        other_domain: Domain,
        num_expanded: int,
    ) -> Optional[tuple[Trajectory, Trajectory]]:
        """
        Returns a trajectory if state is a solution to this problem, None otherwise.
        """
        other_node = self.is_merge_goal(node.state, other_domain)
        if other_node is not None:
            if self.forward:
                f_common_node = node
                b_common_node = other_node
                f_domain = self
                b_domain = other_domain
            else:
                f_common_node = other_node
                b_common_node = node
                f_domain = other_domain
                b_domain = self

            f_traj = from_common_node(
                agent, f_domain, f_common_node, b_common_node, num_expanded
            )
            b_traj = from_common_node(
                agent,
                b_domain,
                b_common_node,
                f_common_node,
                num_expanded,
                b_domain.goal_state_t,
                forward=False,
            )

            return (f_traj, b_traj)
        else:
            return None

    @property
    @abstractmethod
    def num_actions(self) -> int:
        pass

    @property
    @abstractmethod
    def in_channels(self) -> int:
        pass

    @property
    @abstractmethod
    def state_t_width(self) -> int:
        pass

    @property
    @abstractmethod
    def state_t_depth(self) -> int:
        pass

    @abstractmethod
    def is_goal(self, state: State) -> bool:
        pass

    @abstractmethod
    def backward_domain(self) -> Domain:
        pass

    @abstractmethod
    def reverse_action(self, action) -> int:
        pass

    @abstractmethod
    def state_tensor(self, state: State) -> Tensor:
        pass

    @abstractmethod
    def _actions(self, parent_action, state: State) -> list:
        pass

    @abstractmethod
    def _actions_unpruned(self, state: State) -> list:
        pass

    @abstractmethod
    def result(self, state: State, action) -> State:
        pass

    def get_merge_state(self, dir1_state, dir2_parent_state, action) -> State:
        return dir2_parent_state
