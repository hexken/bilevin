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
import time
from typing import TYPE_CHECKING

import torch as to
from torch.nn.functional import log_softmax

from domains.domain import State
from enums import TwoDir
from search.agent import Agent
from search.levin import LevinNode, PriorityQueue, levin_cost, swap_node_contents

if TYPE_CHECKING:
    from domains.domain import Domain, Problem


class BiLevin(Agent):
    @property
    def bidirectional(cls):
        return True

    @property
    def trainable(cls):
        return True

    def search(
        self,
        problem: Problem,
        model,
        budget,
        update_levin_costs=False,
        train=False,
        end_time=None,
    ):
        """ """
        f_frontier = PriorityQueue()
        b_frontier = PriorityQueue()
        f_reached = {}
        b_reached = {}

        problem_id = problem.id
        f_domain = problem.domain

        try_make_solution = f_domain.try_make_solution_func

        b_domain = f_domain.backward_domain()

        f_state = f_domain.reset()
        assert isinstance(f_state, State)
        f_state_t = f_domain.state_tensor(f_state).unsqueeze(0)

        b_states = b_domain.reset()
        if isinstance(b_states, list):
            b_state_t = to.stack([b_domain.state_tensor(s) for s in b_states])
        else:
            b_state_t = b_domain.state_tensor(b_states)
            b_state_t = b_state_t.unsqueeze(0)
            b_states = [b_states]

        feats = model.feature_net(to.vstack((f_state_t, b_state_t)))
        f_state_feats = feats[0]
        b_states_feat = feats[1:]

        b_goal_feats = deepcopy(f_state_feats)

        f_action_logits = model.forward_policy(f_state_feats)
        b_action_logits = model.backward_policy(b_states_feat, b_goal_feats)

        f_log_action_probs = log_softmax(f_action_logits, dim=-1)
        b_log_action_probs = log_softmax(b_action_logits, dim=-1)

        f_start_node = LevinNode(
            f_state,
            g_cost=0,
            log_prob=0.0,
            levin_cost=0.0,
            log_action_probs=f_log_action_probs,
        )
        f_frontier.enqueue(f_start_node)
        f_reached[f_start_node] = f_start_node
        f_domain.update(f_start_node)

        for i, state in enumerate(b_states):
            start_node = LevinNode(
                state,
                g_cost=0,
                log_prob=0.0,
                levin_cost=0.0,
                log_action_probs=b_log_action_probs[i],
            )
            b_frontier.enqueue(start_node)
            b_reached[start_node] = start_node
            b_domain.update(start_node)

        children_to_be_evaluated = []
        state_t_of_children_to_be_evaluated = []

        num_expanded = 0
        num_generated = 0
        while len(f_frontier) > 0 and len(b_frontier) > 0:
            if (
                (budget and num_expanded >= budget)
                or end_time
                and time.time() > end_time
            ):
                return (False, num_expanded, num_generated, None)

            b = b_frontier.top()
            f = f_frontier.top()

            if f < b:
                direction = TwoDir.FORWARD
                _domain = f_domain
                _policy = model.forward_policy
                _frontier = f_frontier
                _reached = f_reached
                other_domain = b_domain
            else:
                direction = TwoDir.BACKWARD
                _domain = b_domain
                _policy = model.backward_policy
                _frontier = b_frontier
                _reached = b_reached
                other_domain = f_domain

            node = _frontier.dequeue()
            num_expanded += 1
            actions = _domain.actions(node.parent_action, node.state)
            if not actions:
                continue

            for a in actions:
                new_state = _domain.result(node.state, a)

                new_node = LevinNode(
                    new_state,
                    g_cost=node.g_cost + 1,
                    parent=node,
                    parent_action=a,
                    log_prob=node.log_prob + node.log_action_probs[a].item(),
                )
                new_node.levin_cost = levin_cost(new_node)
                num_generated += 1

                if new_node not in _reached:
                    trajs = try_make_solution(
                        _domain, new_node, other_domain, num_expanded
                    )

                    if trajs:  # solution found
                        solution_len = len(trajs[0])
                        assert solution_len == len(trajs[1])
                        if not train:
                            trajs = trajs[0]
                        return solution_len, num_expanded, num_generated, trajs

                    _reached[new_node] = new_node
                    _frontier.enqueue(new_node)
                    _domain.update(new_node)

                    children_to_be_evaluated.append(new_node)
                    state_t = _domain.state_tensor(new_state)
                    state_t_of_children_to_be_evaluated.append(state_t)

                elif update_levin_costs:
                    old_node = _reached[new_node]
                    if new_node.g_cost < old_node.g_cost:
                        # print("updating")
                        swap_node_contents(new_node, old_node)
                        if old_node in _frontier:
                            # print("updating frontier")
                            _frontier.remove(old_node)
                            _frontier.enqueue(old_node)

            if children_to_be_evaluated:
                batch_states = to.stack(state_t_of_children_to_be_evaluated)
                batch_feats = model.feature_net(batch_states)
                if direction == TwoDir.BACKWARD:
                    action_logits = _policy(batch_feats, b_goal_feats)
                else:
                    action_logits = _policy(batch_feats)
                log_action_probs = log_softmax(action_logits, dim=-1)

                for i, child in enumerate(children_to_be_evaluated):
                    child.log_action_probs = log_action_probs[i]

                children_to_be_evaluated = []
                state_t_of_children_to_be_evaluated = []

        print(f"Emptied frontiers for problem {problem_id}")
        return False, num_expanded, num_generated, None
