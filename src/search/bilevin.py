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
from copy import deepcopy
import heapq
import random
import time
from typing import TYPE_CHECKING

import torch as to
from torch.jit import RecursiveScriptModule
from torch.nn.functional import log_softmax

from domains.domain import State
from enums import TwoDir
from search.agent import Agent
from search.utils import LevinNode, levin_cost

if TYPE_CHECKING:
    from domains.domain import Domain, Problem


class BiLevin(Agent):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

    @property
    def bidirectional(cls):
        return True

    @property
    def trainable(cls):
        return True

    def search(
        self,
        problem: Problem,
        budget,
        train=False,
        end_time=None,
    ):
        """ """
        f_frontier = []
        b_frontier = []
        f_reached = {}
        b_reached = {}

        model = self.model

        problem_id = problem.id
        f_domain = problem.domain
        num_actions = f_domain.num_actions

        try_make_solution = f_domain.try_make_solution_func

        f_state = f_domain.reset()
        assert isinstance(f_state, State)
        f_state_t = f_domain.state_tensor(f_state).unsqueeze(0)
        f_actions, f_mask = f_domain.actions_unpruned(f_state)
        f_log_probs = model(f_state_t, mask=f_mask)
        f_start_node = LevinNode(
            f_state,
            g_cost=0,
            log_prob=0.0,
            levin_cost=0.0,
            actions=f_actions,
            log_action_probs=f_log_probs,
        )
        f_reached[f_start_node] = f_start_node
        f_domain.update(f_start_node)
        if f_actions:
            f_frontier.append(f_start_node)
        heapq.heapify(f_frontier)

        b_domain = f_domain.backward_domain()
        if b_domain.requires_backward_goal:
            b_goal_feats = model.backward_feature_net(f_state_t)
        else:
            b_goal_feats = None

        b_states = b_domain.reset()
        if isinstance(b_states, list):
            b_state_t = []
            b_actions = []
            b_mask = []
            for i, s in enumerate(b_states):
                b_state_t.append(b_domain.state_tensor(s))
                actions, mask = b_domain.actions_unpruned(s)
                b_actions.append(actions)
                b_mask.append(mask)
            b_state_t = to.stack(b_state_t)
        else:
            b_state_t = b_domain.state_tensor(b_states)
            b_state_t = b_state_t.unsqueeze(0)
            b_actions, b_mask = b_domain.actions_unpruned(b_states)
            b_states = [b_states]
            b_actions = [b_actions]

        b_log_probs = model(b_state_t, forward=False, goal_feats=f_state_t, mask=b_mask)
        for i, state in enumerate(b_states):
            start_node = LevinNode(
                state,
                g_cost=0,
                log_prob=0.0,
                levin_cost=0.0,
                actions=b_actions[i],
                log_action_probs=b_log_probs[i],
            )
            b_reached[start_node] = start_node
            b_domain.update(start_node)
            if start_node.actions:
                b_frontier.append(start_node)
        heapq.heapify(b_frontier)

        n_total_expanded = 0
        n_forw_expanded = 0
        n_backw_expanded = 0
        n_forw_generated = 0
        n_backw_generated = 0

        while len(f_frontier) > 0 and len(b_frontier) > 0:
            if (
                (budget and n_total_expanded >= budget)
                or end_time
                and time.time() > end_time
            ):
                return (
                    False,
                    n_forw_expanded,
                    n_backw_expanded,
                    n_forw_generated,
                    n_backw_generated,
                    None,
                )

            if f_frontier[0] < b_frontier[0]:
                direction = TwoDir.FORWARD
                _goal_feats = None
                _domain = f_domain
                _frontier = f_frontier
                _reached = f_reached
                other_domain = b_domain
            else:
                direction = TwoDir.BACKWARD
                _domain = b_domain
                _goal_feats = b_goal_feats
                _frontier = b_frontier
                _reached = b_reached
                other_domain = f_domain

            node = heapq.heappop(_frontier)
            if direction == TwoDir.FORWARD:
                n_forw_expanded += 1
            else:
                n_backw_expanded += 1
            n_total_expanded += 1

            masks = []
            children_to_be_evaluated = []
            state_t_of_children_to_be_evaluated = []
            for a in node.actions:
                new_state = _domain.result(node.state, a)
                new_state_actions, mask = _domain.actions(a, new_state)

                new_node = LevinNode(
                    new_state,
                    g_cost=node.g_cost + 1,
                    parent=node,
                    parent_action=a,
                    actions=new_state_actions,
                    log_prob=node.log_prob + node.log_action_probs[a].item(),
                )
                new_node.levin_cost = levin_cost(new_node)

                if direction == TwoDir.FORWARD:
                    n_forw_generated += 1
                else:
                    n_backw_generated += 1

                if new_node not in _reached:
                    trajs = try_make_solution(
                        model,
                        _domain,
                        new_node,
                        other_domain,
                        n_forw_expanded + n_backw_expanded,
                    )

                    if trajs:  # solution found
                        return (
                            len(trajs[0]),
                            n_forw_expanded,
                            n_backw_expanded,
                            n_forw_generated,
                            n_backw_generated,
                            trajs,
                        )

                    _reached[new_node] = new_node
                    _domain.update(new_node)

                    if new_state_actions:
                        heapq.heappush(_frontier, new_node)
                        children_to_be_evaluated.append(new_node)
                        state_t = _domain.state_tensor(new_state)
                        state_t_of_children_to_be_evaluated.append(state_t)
                        masks.append(mask)

            if children_to_be_evaluated:
                children_state_t = to.stack(state_t_of_children_to_be_evaluated)
                masks = to.stack(masks)
                log_probs = model(
                    children_state_t,
                    forward=direction == TwoDir.FORWARD,
                    goal_feats=_goal_feats,
                    masks=masks,
                )

                for i, child in enumerate(children_to_be_evaluated):
                    child.log_action_probs = log_probs[i]

        print(f"Emptied frontiers for problem {problem_id}")
        return (
            False,
            n_forw_expanded,
            n_backw_expanded,
            n_forw_generated,
            n_backw_generated,
            None,
        )
