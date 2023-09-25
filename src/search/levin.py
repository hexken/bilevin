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
from typing import TYPE_CHECKING

import torch as to
from torch.jit import RecursiveScriptModule

from domains.domain import State
from search.agent import Agent
from search.utils import SearchNode, Problem, Trajectory


class Levin(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def bidirectional(cls):
        return False

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

        problem_id = problem.id
        domain = problem.domain
        model = self.model
        cost_fn = self.cost_fn

        state = domain.reset()
        assert isinstance(state, State)
        state_t = domain.state_tensor(state).unsqueeze(0)
        actions, mask = domain.actions_unpruned(state)
        log_probs, _ = model(state_t, mask=mask)

        node = SearchNode(
            state,
            g_cost=0,
            log_prob=0.0,
            cost=0.0,
            actions=actions,
            actions_mask=mask,
            log_action_probs=log_probs[0],
        )

        reached = {node: node}
        frontier = []
        if actions:
            frontier.append(node)
        heapq.heapify(frontier)

        num_expanded = 0
        num_generated = 0
        while len(frontier) > 0:
            if (
                (exp_budget > 0 and num_expanded >= exp_budget)
                or time_budget > 0
                and timer() - start_time >= time_budget
            ):
                return num_expanded, 0, num_generated, 0, None

            node = heapq.heappop(frontier)
            num_expanded += 1

            masks = []
            children_to_be_evaluated = []
            state_t_of_children_to_be_evaluated = []
            for a in node.actions:
                new_state = domain.result(node.state, a)
                new_state_actions, mask = domain.actions(a, new_state)

                new_node = SearchNode(
                    new_state,
                    g_cost=node.g_cost + 1,
                    parent=node,
                    parent_action=a,
                    actions=new_state_actions,
                    actions_mask=mask,
                    log_prob=node.log_prob + node.log_action_probs[a].item(),
                )
                new_node.cost = cost_fn(new_node)

                num_generated += 1

                if new_node not in reached:
                    if domain.is_goal(new_state):
                        traj = Trajectory.from_goal_node(
                            domain=domain,
                            final_node=new_node,
                            num_expanded=num_expanded,
                            partial_g_cost=new_node.g_cost,
                            partial_log_prob=new_node.log_prob,
                            log_prob=new_node.log_prob,
                            model=model,
                        )
                        traj = (traj, None)
                        return num_expanded, 0, num_generated, 0, traj

                    reached[new_node] = new_node
                    if new_state_actions:
                        heapq.heappush(frontier, new_node)
                        children_to_be_evaluated.append(new_node)
                        state_t = domain.state_tensor(new_state)
                        state_t_of_children_to_be_evaluated.append(state_t)
                        masks.append(mask)

            if children_to_be_evaluated:
                children_state_t = to.stack(state_t_of_children_to_be_evaluated)
                masks = to.stack(masks)
                log_probs, _ = model(
                    children_state_t,
                    mask=masks,
                )

                for child, lap in zip(children_to_be_evaluated, log_probs):
                    child.log_action_probs = lap

        print(f"Emptied frontier for problem {problem_id}")
        return num_expanded, 0, num_generated, 0, None
