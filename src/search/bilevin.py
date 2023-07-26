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
import heapq
from timeit import default_timer as timer

import torch as to

from domains.domain import State
from enums import TwoDir
from search.agent import Agent
from search.utils import LevinNode, Problem


class BiLevin(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def bidirectional(cls):
        return True

    @property
    def trainable(cls):
        return True

    def search(
        self,
        problem: Problem,
        exp_budget: int,
        time_budget: float,
    ):
        """ """
        start_time = timer()

        f_frontier = []
        b_frontier = []
        f_reached = {}
        b_reached = {}

        model = self.model
        cost_fn = self.cost_fn

        problem_id = problem.id
        f_domain = problem.domain
        b_domain = f_domain.backward_domain()

        f_state = f_domain.reset()
        assert isinstance(f_state, State)
        f_state_t = f_domain.state_tensor(f_state).unsqueeze(0)
        f_actions, f_mask = f_domain.actions_unpruned(f_state)
        f_log_probs, _ = model(f_state_t, mask=f_mask)
        f_start_node = LevinNode(
            f_state,
            g_cost=0,
            log_prob=0.0,
            levin_cost=0.0,
            actions=f_actions,
            actions_mask=f_mask,
            log_action_probs=f_log_probs[0],
        )
        f_reached[f_start_node] = f_start_node
        f_domain.update(f_start_node)
        if f_actions:
            f_frontier.append(f_start_node)
        heapq.heapify(f_frontier)

        if model.backward_goal:
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
            b_mask = to.stack(b_mask)
            b_state_t = to.stack(b_state_t)
        else:
            b_state_t = b_domain.state_tensor(b_states)
            b_state_t = b_state_t.unsqueeze(0)
            b_actions, b_mask = b_domain.actions_unpruned(b_states)
            b_states = [b_states]
            b_actions = [b_actions]
            b_mask = b_mask.unsqueeze(0)

        b_log_probs, _ = model(
            b_state_t, forward=False, goal_feats=b_goal_feats, mask=b_mask
        )
        for i, state in enumerate(b_states):
            start_node = LevinNode(
                state,
                g_cost=0,
                log_prob=0.0,
                levin_cost=0.0,
                actions=b_actions[i],
                actions_mask=b_mask[i],
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
                (exp_budget > 0 and n_total_expanded >= exp_budget)
                or time_budget > 0
                and timer() - start_time >= time_budget
            ):
                return (
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
                    actions_mask=mask,
                    log_prob=node.log_prob + node.log_action_probs[a].item(),
                )
                new_node.levin_cost = cost_fn(new_node)

                if direction == TwoDir.FORWARD:
                    n_forw_generated += 1
                else:
                    n_backw_generated += 1

                if new_node not in _reached:
                    trajs = _domain.try_make_solution(
                        model,
                        new_node,
                        other_domain,
                        n_forw_expanded + n_backw_expanded,
                    )

                    if trajs:  # solution found
                        return (
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
                log_probs, _ = model(
                    children_state_t,
                    forward=direction == TwoDir.FORWARD,
                    goal_feats=_goal_feats,
                    mask=masks,
                )

                for child, lap in zip(children_to_be_evaluated, log_probs):
                    child.log_action_probs = lap

        print(f"Emptied frontiers for problem {problem_id}")
        return (
            n_forw_expanded,
            n_backw_expanded,
            n_forw_generated,
            n_backw_generated,
            None,
        )
