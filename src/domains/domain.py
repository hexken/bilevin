from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Type

import torch as to
from torch import Tensor, full

from models import AgentModel
from search.utils import SearchNode, Trajectory


class State(ABC):
    @abstractmethod
    def __eq__(self) -> bool:
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

    def reset(self) -> State:
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

    def try_make_solution(
        self,
        model: AgentModel,
        node: SearchNode,
        other_domain: Domain,
        num_expanded: int,
    ) -> Optional[tuple[Trajectory, Trajectory]]:
        """
        Returns a trajectory if state is a solution to this problem, None otherwise.
        """
        hsh = node.state.__hash__()
        if hsh in other_domain.aux_closed:  # solution found
            other_node = other_domain.aux_closed[hsh]
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

            f_traj = get_merged_trajectory(
                model, f_domain, f_common_node, b_common_node, type(node), num_expanded
            )
            b_traj = get_merged_trajectory(
                model,
                b_domain,
                b_common_node,
                f_common_node,
                type(node),
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
    def state_width(self) -> int:
        pass

    @property
    @abstractmethod
    def requires_backward_goal(self) -> bool:
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


def get_merged_trajectory(
    model: AgentModel,
    dir1_domain: Domain,
    dir1_common: SearchNode,
    dir2_common: SearchNode,
    node_type: Type[SearchNode],
    num_expanded: int,
    goal_state_t: Optional[Tensor] = None,
    forward: bool = True,
):
    """
    Returns a new trajectory going from dir1_start to dir2_start, passing through
    merge(dir1_common, dir2_common).
    """
    dir1_node = dir1_common

    dir2_parent_node = dir2_common.parent
    dir2_parent_action = dir2_common.parent_action

    while dir2_parent_node:
        new_state = dir2_parent_node.state
        actions, mask = dir1_domain.actions_unpruned(new_state)
        new_dir1_node = node_type(
            state=new_state,
            g_cost=dir1_node.g_cost + 1,
            parent=dir1_node,
            parent_action=dir1_domain.reverse_action(dir2_parent_action),
            actions=actions,
            actions_mask=mask,
        )
        dir1_node = new_dir1_node
        dir2_parent_action = dir2_parent_node.parent_action
        dir2_parent_node = dir2_parent_node.parent

    return Trajectory.from_goal_node(
        domain=dir1_domain,
        final_node=dir1_node,
        num_expanded=num_expanded,
        partial_g_cost=dir1_common.g_cost,
        partial_log_prob=dir1_common.log_prob,
        goal_state_t=goal_state_t,
        forward=forward,
    )
