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
from torch.nn.functional import cross_entropy, nll_loss

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
        log_action_probs: Optional[to.Tensor] = None,
    ):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.g_cost = g_cost
        self.log_prob = log_prob
        self.actions = actions
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
        log_action_probs: Optional[to.Tensor] = None,
        levin_cost: Optional[float] = None,
    ):
        super().__init__(
            state=state,
            g_cost=g_cost,
            parent=parent,
            parent_action=parent_action,
            log_prob=log_prob,
            actions=actions,
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
        model: AgentModel,
        domain: Domain,
        final_node: SearchNode,
        num_expanded: int,
        partial_g_cost: int,  # g_cost of node that generated sol.
        partial_log_prob: float,  # probability of node that generates sol.
        goal_state_t: Optional[to.Tensor] = None,
        forward: bool = True,
    ):
        """
        Receives a SearchNode representing a solution to the problem.
        Backtracks the path performed by search, collecting state-action pairs along the way.
        """
        self.forward = forward
        self.partial_g_cost = partial_g_cost
        self.partial_log_prob = partial_log_prob
        self.num_expanded = num_expanded
        self.goal_state_t: Optional[to.Tensor] = (
            goal_state_t.unsqueeze(0) if goal_state_t is not None else None
        )

        action = final_node.parent_action
        node = final_node.parent
        states = []
        actions = []
        masks = []

        while node:
            state_t = domain.state_tensor(node.state)
            states.append(state_t)
            actions.append(action)
            masks.append(node.actions)
            action = node.parent_action
            node = node.parent

        self.states = to.stack(tuple(reversed(states)))
        self.actions = to.tensor(tuple(reversed(actions)))
        # print([len(m) for m in masks[::-1]])
        rows = to.repeat_interleave(
            to.arange(len(masks)),
            to.tensor(tuple(len(m) for m in reversed(masks))),
        )
        cols = to.tensor(tuple(j for m in reversed(masks) for j in m))
        self.masks = to.zeros(len(masks), domain.num_actions)
        self.masks[rows, cols] = 1
        self.cost_to_gos = to.arange(len(self.states), 0, -1)

        self.log_prob, pnll = traj_nll(
            self, model
        )  # do this after states, actions, forward, steps, are set

        # assert np.isclose(-1 * self.partial_log_prob, pnll)
        if not np.isclose(-1 * self.partial_log_prob, pnll):
            print(
                f"Warning: partial log prob does not match nll {-1 * self.partial_log_prob} {pnll}"
            )

    def __len__(self):
        return len(self.states)


def traj_nll(traj: Trajectory, model: AgentModel):
    with to.no_grad():
        state_feats = model.feature_net(traj.states)

        if traj.forward:
            logits = model.forward_policy(state_feats)
        else:
            if traj.goal_state_t is not None:
                goal_feat = model.feature_net(traj.goal_state_t)
                logits = model.backward_policy(state_feats, goal_feat)
            else:
                logits = model.backward_policy(state_feats)

        action_log_probs = masked_log_softmax(logits, traj.masks)
        action_nlls = nll_loss(action_log_probs, traj.actions, reduction="none")
        traj_nll = action_nlls.sum().item()
        traj_partial_nll = action_nlls[: traj.partial_g_cost].sum().item()

    # might be shifted? masking causing issues?
    return traj_nll, traj_partial_nll


class MergedTrajectory:
    def __init__(self, trajs: list[Trajectory]):
        self.forward = False
        if trajs:
            if trajs[0].forward:
                self.forward = True
                self.goal_states = None
            else:
                self.goal_states = to.cat(tuple(t.goal_state_t for t in trajs))

            self.states = to.cat(tuple(t.states for t in trajs))
            self.actions = to.cat(tuple(t.actions for t in trajs))
            self.lengths = to.tensor(tuple(len(t) for t in trajs))
            self.steps = tuple(t.partial_g_cost for t in trajs)

            indices = to.arange(len(trajs))
            self.indices = to.repeat_interleave(indices, self.lengths)
            self.nums_expanded = to.tensor(
                tuple(t.num_expanded for t in trajs), dtype=to.float32
            )
            self.num_trajs = len(self.nums_expanded)
            self.num_states = len(self.states)

            # if shuffle:
            #     self.shuffle()
        else:
            return None

    def __len__(self):
        raise NotImplementedError

    # def shuffle(self):
    #     perm = to.randperm(self.num_states)
    #     self.states = self.states[perm]
    #     self.actions = self.actions[perm]
    #     self.indices = self.indices[perm]


def get_merged_solution(
    model: AgentModel,
    dir1_domain: Domain,
    dir1_common: SearchNode,
    dir2_common: SearchNode,
    node_type: Type[SearchNode],
    num_expanded: int,
    goal_state_t: Optional[to.Tensor] = None,
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
        new_dir1_node = node_type(
            state=state,
            g_cost=dir1_node.g_cost + 1,
            parent=dir1_node,
            parent_action=dir1_domain.reverse_action(parent_dir2_action),
            actions=dir1_domain.actions_unpruned(state),
        )
        dir1_node = new_dir1_node
        parent_dir2_action = parent_dir2_node.parent_action
        parent_dir2_node = parent_dir2_node.parent

    if dir1_common.parent:
        log_prob = dir1_common.parent.log_prob
    else:
        log_prob = 0.0

    return Trajectory(
        model=model,
        domain=dir1_domain,
        final_node=dir1_node,
        num_expanded=num_expanded,
        partial_g_cost=dir1_common.g_cost - 1,
        partial_log_prob=log_prob,
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

        f_traj = get_merged_solution(
            model, f_domain, f_common_node, b_common_node, type(node), num_expanded
        )
        b_traj = get_merged_solution(
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


"""
Originally from https://github.com/allenai/allennlp/blob/b6cc9d39651273e8ec2a7e334908ffa9de5c2026/allennlp/nn/util.py#L272-L303
"""


def masked_log_softmax(vector: to.Tensor, mask: to.Tensor, dim: int = -1) -> to.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        vector = vector + (mask + 1e-45).log()
    return to.nn.functional.log_softmax(vector, dim=dim)
