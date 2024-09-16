from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, Optional, TYPE_CHECKING

from torch import Tensor

from domains.state import TState
from search.traj import Trajectory

if TYPE_CHECKING:
    from search.node import SearchNode
    from search.agent import Agent


class Domain(ABC, Generic[TState]):
    def __init__(self, forward: bool = True):
        self.aux_closed: dict = {}
        self.forward: bool = forward
        self.start_state: TState
        self.goal_state_t: Optional[Tensor] = None

    @abstractmethod
    def init(self) -> TState:
        pass

    def _init(self) -> TState | list[TState]:
        del self.aux_closed
        self.aux_closed = {}
        return self.start_state

    def update(self, node: SearchNode):
        self.aux_closed[node.state.__hash__()] = node

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
    ) -> tuple[Trajectory | None, Trajectory | None]:
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

            f_traj = Trajectory.from_common_node(
                agent,
                f_domain,
                f_common_node,
                b_common_node,
                num_expanded,
                forward=True,
                set_masks=agent.mask_invalid_actions,
            )
            b_traj = Trajectory.from_common_node(
                agent,
                b_domain,
                b_common_node,
                f_common_node,
                num_expanded,
                b_domain.goal_state_t,
                forward=False,
                set_masks=agent.mask_invalid_actions,
            )

            return (f_traj, b_traj)
        else:
            return (None, None)

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
    def is_goal(self, state: TState) -> bool:
        pass

    @abstractmethod
    def backward_domain(self) -> Domain:
        pass

    @abstractmethod
    def reverse_action(self, action) -> int:
        pass

    @abstractmethod
    def state_tensor(self, state: TState) -> Tensor:
        pass

    @abstractmethod
    def actions(self, parent_action, state: TState) -> list:
        pass

    @abstractmethod
    def actions_unpruned(self, state: TState) -> list:
        pass

    @abstractmethod
    def result(self, state: TState, action) -> TState:
        pass

    def get_merge_state(self, dir1_state, dir2_parent_state, action) -> TState:
        """
        Returns the state that results from applying action to dir1_state, assuming dir1 and dir2
        nodes are currently the same state.
        """
        return dir2_parent_state
