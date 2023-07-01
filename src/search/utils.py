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
import math
from typing import Optional, Type

import numpy as np
import torch as to
from torch import Tensor
from torch.nn.functional import nll_loss

from domains.domain import Domain, State
from models.models import AgentModel

search_result_header = [
    "ProblemId",
    "Time",
    "Exp",
    "FExp",
    "BExp",
    "Gen",
    "FGen",
    "BGen",
    "Len",
    "Fg",
    "Bg",
    "FPnll",
    "BPnll",
    "Fnll",
    "Bnll",
    "EndTime",
]
int_columns = ["Exp", "FExp", "BExp", "Gen", "FGen", "BGen"]


class SearchNode:
    def __init__(
        self,
        state: State,
        g_cost: int,
        parent: Optional[SearchNode] = None,
        parent_action: Optional[int] = None,
        log_prob: Optional[float] = None,
        actions: Optional[list[int]] = None,
        actions_mask: Optional[Tensor] = None,
        log_action_probs: Optional[Tensor] = None,
    ):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.g_cost = g_cost
        self.log_prob = log_prob
        self.actions = actions
        self.actions_mask = actions_mask
        self.log_action_probs = log_action_probs

    def __eq__(self, other):
        """
        Verify if two SearchNodes are identical by verifying the
         state in the nodes.
        """
        return self.state == other.state

    def __lt__(self, other):
        """
        less-than used by the heap
        """
        return self.g_cost < other.g_cost

    def __hash__(self):
        """
        Hash function used in the closed list
        """
        return self.state.__hash__()


class LevinNode(SearchNode):
    def __init__(
        self,
        state: State,
        g_cost: int,
        parent: Optional[SearchNode] = None,
        parent_action: Optional[int] = None,
        log_prob: Optional[float] = None,
        actions: Optional[list[int]] = None,
        actions_mask: Optional[Tensor] = None,
        log_action_probs: Optional[Tensor] = None,
        levin_cost: Optional[float] = None,
    ):
        super().__init__(
            state=state,
            g_cost=g_cost,
            parent=parent,
            parent_action=parent_action,
            log_prob=log_prob,
            actions=actions,
            actions_mask=actions_mask,
            log_action_probs=log_action_probs,
        )
        self.levin_cost = levin_cost

    def __lt__(self, other):
        """
        used by the heap
        """
        return self.levin_cost < other.levin_cost


def levin_cost(node: LevinNode):
    return math.log(node.g_cost) - node.log_prob  # type:ignore


class Trajectory:
    def __init__(
        self,
        states: Tensor,
        actions: Tensor,
        masks: Tensor,
        num_expanded: int,
        partial_g_cost: Optional[int] = None,  # g_cost of node that generated sol.
        partial_log_prob: Optional[
            float
        ] = None,  # probability of node that generates sol.
        log_prob: Optional[float] = None,
        goal_state_t: Optional[Tensor] = None,
        forward: bool = True,
    ):
        self.states = states
        self.actions = actions
        self.num_expanded = num_expanded
        self.partial_g_cost = partial_g_cost
        self.partial_log_prob = partial_log_prob
        self.log_prob = log_prob
        self.masks = masks
        self.goal_state_t = goal_state_t
        self.forward = forward

        self.cost = len(self.actions)
        self.costs_to_go = to.arange(self.cost, 0, -1)
        self._len = len(self.actions)

    @classmethod
    def from_goal_node(
        cls,
        domain: Domain,
        final_node: SearchNode,
        num_expanded: int,
        partial_g_cost: int,
        partial_log_prob: Optional[float] = None,
        log_prob: Optional[float] = None,
        model: Optional[AgentModel] = None,
        goal_state_t: Optional[Tensor] = None,
        forward: bool = True,
    ):
        """
        Receives a SearchNode representing a solution to the problem.
        Backtracks the path performed by search, collecting state-action pairs along the way.
        """
        goal_state_t = goal_state_t.unsqueeze(0) if goal_state_t is not None else None
        action = final_node.parent_action
        node = final_node.parent

        states = []
        actions = []
        masks = []

        while node:
            state_t = domain.state_tensor(node.state)
            states.append(state_t)
            actions.append(action)
            masks.append(node.actions_mask)
            action = node.parent_action
            node = node.parent

        states = to.stack(tuple(reversed(states)))
        actions = to.tensor(tuple(reversed(actions)))
        masks = to.stack(tuple(reversed(masks)))

        if model:
            with to.no_grad():
                k = partial_g_cost
                log_probs, _ = model(
                    states,
                    mask=masks,
                    forward=forward,
                    goal_state_t=goal_state_t,
                )
                partial_nll = nll_loss(
                    log_probs[:k], actions[:k], reduction="sum"
                ).item()
                nll = (
                    partial_nll
                    + nll_loss(log_probs[k:], actions[k:], reduction="sum").item()
                )

            lp = -nll
            plp = -partial_nll

            # todo remove this once we're sure it's working
            if log_prob and not np.isclose(log_prob, lp):
                print(f"Warning: search log_prob != model log_prob {log_prob} {lp}")
            else:
                log_prob = lp
            if partial_log_prob and not np.isclose(partial_log_prob, plp):
                print(
                    f"Warning: search partial_log_prob != model partial_log_prob {partial_log_prob} {plp}"
                )
            else:
                partial_log_prob = plp

        return cls(
            states=states,
            actions=actions,
            masks=masks,
            num_expanded=num_expanded,
            partial_g_cost=partial_g_cost,
            partial_log_prob=partial_log_prob,
            log_prob=log_prob,
            goal_state_t=goal_state_t,
            forward=forward,
        )

    def __len__(self):
        return self._len

    def get_subgoal_trajs(self: Trajectory):
        """
        Generates all sub-trajectories of a trajectory.
        """
        sub_trajs = []
        for goal_idx in range(len(self) - 1, 0, -1):
            masks = self.masks[:goal_idx]

            est_num_exp = int(self.num_expanded / (len(self) - goal_idx + 1))
            sub_trajs.append(
                Trajectory(
                    self.states[:goal_idx],
                    self.actions[:goal_idx],
                    masks=masks,
                    num_expanded=est_num_exp,
                    forward=False,
                    goal_state_t=self.states[goal_idx].unsqueeze(0),
                )
            )
        return sub_trajs


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

    parent_dir2_node = dir2_common.parent
    parent_dir2_action = dir2_common.parent_action

    while parent_dir2_node:
        state = parent_dir2_node.state
        actions, mask = dir1_domain.actions_unpruned(state)
        new_dir1_node = node_type(
            state=state,
            g_cost=dir1_node.g_cost + 1,
            parent=dir1_node,
            parent_action=dir1_domain.reverse_action(parent_dir2_action),
            actions=actions,
            actions_mask=mask,
        )
        dir1_node = new_dir1_node
        parent_dir2_action = parent_dir2_node.parent_action
        parent_dir2_node = parent_dir2_node.parent

    return Trajectory.from_goal_node(
        domain=dir1_domain,
        final_node=dir1_node,
        num_expanded=num_expanded,
        partial_g_cost=dir1_common.g_cost,
        partial_log_prob=dir1_common.log_prob,
        model=model,  # todo remove this after debugging ?
        goal_state_t=goal_state_t,
        forward=forward,
    )


def try_make_solution(
    model: AgentModel,
    this_domain: Domain,
    node: SearchNode,
    other_domain: Domain,
    num_expanded: int,
) -> Optional[tuple[Trajectory, Trajectory]]:
    """
    Returns a trajectory if state is a solution to this problem, None otherwise.
    """
    hsh = node.state.__hash__()
    if hsh in other_domain.visited:  # solution found
        other_node = other_domain.visited[hsh]
        if this_domain.forward:
            f_common_node = node
            b_common_node = other_node
            f_domain = this_domain
            b_domain = other_domain
        else:
            f_common_node = other_node
            b_common_node = node
            f_domain = other_domain
            b_domain = this_domain

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
