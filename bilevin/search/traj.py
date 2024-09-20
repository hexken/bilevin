from __future__ import annotations
from typing import TYPE_CHECKING

from numpy import nan
import torch as to
from torch import Tensor
from torch.nn.functional import nll_loss

from search.node import SearchNode


if TYPE_CHECKING:
    from search.agent import Agent
    from domains.domain import Domain


class Trajectory:
    def __init__(
        self,
        states: Tensor,
        actions: Tensor,
        masks: Tensor | None,
        num_expanded: int,
        partial_g_cost: int,  # g_cost of node that generated sol.
        goal_state_t: Tensor | None = None,
        forward: bool = True,
    ):
        self.states = states
        self.actions = actions
        self.num_expanded = num_expanded
        self.partial_g_cost = partial_g_cost
        self.masks = masks
        self.goal_state_t = goal_state_t
        self.forward = forward

        self._len = len(self.actions)

    def __len__(self):
        return self._len

    # def dict(self):
    #     return {
    #         key: value
    #         for key, value in self.__dict__.items()
    #         if not key.startswith("__") and not callable(value)
    #     }

    @classmethod
    def from_common_node(
        cls,
        agent: Agent,
        dir1_domain: Domain,
        dir1_common: SearchNode,
        dir2_common: SearchNode,
        num_expanded: int,
        goal_state_t: Tensor | None = None,
        forward: bool = True,
        set_masks: bool = False,
    ) -> Trajectory:
        """
        Returns a new trajectory going from dir1_start to dir2_start, passing through
        merge(dir1_common, dir2_common).
        """
        dir1_node = dir1_common

        dir2_parent_node = dir2_common.parent
        dir2_parent_action = dir2_common.parent_action

        while dir2_parent_node:
            action = dir1_domain.reverse_action(dir2_parent_action)
            new_state = dir1_domain.get_merge_state(
                dir1_node.state, dir2_parent_node.state, action
            )
            actions = dir1_domain.actions(action, new_state)
            if set_masks:
                mask = agent.get_mask(actions)
            else:
                mask = None
            new_dir1_node = SearchNode(
                state=new_state,
                g=dir1_node.g + 1,
                parent=dir1_node,
                parent_action=action,
                actions=actions,
                mask=mask,
                log_prob=0.0,
            )
            dir1_node = new_dir1_node
            dir2_parent_action = dir2_parent_node.parent_action
            dir2_parent_node = dir2_parent_node.parent

        return Trajectory.from_goal_node(
            agent,
            domain=dir1_domain,
            goal_node=dir1_node,
            num_expanded=num_expanded,
            partial_g_cost=dir1_common.g,
            goal_state_t=goal_state_t,
            forward=forward,
            set_masks=set_masks,
        )

    @classmethod
    def from_goal_node(
        cls,
        agent: Agent,
        domain: Domain,
        goal_node: SearchNode,
        num_expanded: int,
        partial_g_cost: int,
        goal_state_t: Tensor | None,
        forward: bool = True,
        set_masks: bool = False,
    ) -> Trajectory:
        """
        Receives a SearchNode representing a solution to the problem.
        Backtracks the path performed by search, collecting state-action pairs along the way.

        actions[i] is the action taken in state[i] to get to state[i+1]
        """
        assert domain.is_goal(goal_node.state)
        action = goal_node.parent_action
        node = goal_node.parent

        states = []
        actions = []
        masks = []

        while node is not None:
            state_t = domain.state_tensor(node.state)
            states.append(state_t)
            actions.append(action)
            masks.append(node.mask)
            action = node.parent_action
            node = node.parent

        states = to.stack(tuple(reversed(states)))
        actions = to.tensor(tuple(reversed(actions)))
        if set_masks:
            masks = to.stack(tuple(reversed(masks)))
        else:
            masks = None

        return Trajectory(
            states=states,
            actions=actions,
            masks=masks,
            num_expanded=num_expanded,
            partial_g_cost=partial_g_cost,
            goal_state_t=goal_state_t,
            forward=forward,
        )
