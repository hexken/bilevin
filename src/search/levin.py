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
import math
import time
from typing import Optional, TYPE_CHECKING

import torch as to
from torch.jit import RecursiveScriptModule
from torch.nn.functional import log_softmax

from domains.domain import State
from models.loss_functions import masked_log_softmax
from search.agent import Agent
from search.utils import LevinNode, SearchNode, Trajectory, levin_cost

if TYPE_CHECKING:
    from domains.domain import State, Domain, Problem


class Levin(Agent):
    @property
    def bidirectional(cls):
        return False

    @property
    def trainable(cls):
        return True

    def search(
        self,
        problem: Problem,
        budget,
        update_levin_costs=False,
        train=False,
        end_time=None,
        random_goal=False,
    ):
        """ """

        problem_id = problem.id
        domain = problem.domain
        model = self.model
        num_actions = domain.num_actions
        feature_net: RecursiveScriptModule = model.feature_net
        forward_policy: RecursiveScriptModule = model.forward_policy

        state = domain.reset()
        assert isinstance(state, State)
        state_t = domain.state_tensor(state).unsqueeze(0)

        state_feats = feature_net(state_t)
        action_logits = forward_policy(state_feats)

        avail_actions = domain.actions_unpruned(state)
        mask = to.zeros(num_actions)
        mask[avail_actions] = 1
        log_action_probs = masked_log_softmax(action_logits[0], mask, dim=-1)

        node = LevinNode(
            state,
            g_cost=0,
            log_prob=0.0,
            levin_cost=0.0,
            actions=avail_actions,
            log_action_probs=log_action_probs,
        )

        frontier = []
        reached = {}
        if avail_actions:
            frontier.append(node)
        reached[node] = node
        heapq.heapify(frontier)

        num_expanded = 0
        num_generated = 0
        while len(frontier) > 0:
            if (
                (budget and num_expanded >= budget)
                or end_time
                and time.time() > end_time
            ):
                return 0, num_expanded, num_generated, None

            node = heapq.heappop(frontier)
            num_expanded += 1

            masks = []
            children_to_be_evaluated = []
            state_t_of_children_to_be_evaluated = []
            for a in node.actions:
                new_state = domain.result(node.state, a)
                new_state_actions = domain.actions(a, new_state)

                new_node = LevinNode(
                    new_state,
                    g_cost=node.g_cost + 1,
                    parent=node,
                    parent_action=a,
                    actions=new_state_actions,
                    log_prob=node.log_prob + node.log_action_probs[a].item(),
                )
                new_node.levin_cost = levin_cost(new_node)

                num_generated += 1

                if new_node not in reached:
                    if domain.is_goal(new_state):
                        solution_len = new_node.g_cost
                        traj = Trajectory(
                            domain=domain,
                            final_node=new_node,
                            num_expanded=num_expanded,
                            steps=new_node.g_cost - 1,
                            partial_log_prob=new_node.log_prob,
                        )
                        if train:
                            traj = (traj,)
                        return solution_len, num_expanded, num_generated, traj

                    reached[new_node] = new_node
                    if new_state_actions:
                        heapq.heappush(frontier, new_node)

                        children_to_be_evaluated.append(new_node)
                        state_t = domain.state_tensor(new_state)
                        state_t_of_children_to_be_evaluated.append(state_t)

                        mask = to.zeros(num_actions)
                        mask[new_state_actions] = 1
                        masks.append(mask)

            if children_to_be_evaluated:
                batch_states = to.stack(state_t_of_children_to_be_evaluated)
                batch_feats = feature_net(batch_states)
                action_logits = forward_policy(batch_feats)
                masks = to.stack(masks)
                log_action_probs = masked_log_softmax(action_logits, masks, dim=-1)

                for child, lap in zip(children_to_be_evaluated, log_action_probs):
                    child.log_action_probs = lap

        print(f"Emptied frontier for problem {problem_id}")
        return 0, num_expanded, num_generated, None
