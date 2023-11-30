from __future__ import annotations
import heapq
from timeit import default_timer as timer

import numpy as np
import torch as to

from enums import TwoDir
from search.agent import Agent
from search.utils import Problem, Trajectory


class UniDir(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_bidirectional(self):
        return False

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

        state = domain.reset()
        state_t = domain.state_tensor(state).unsqueeze(0)
        actions, mask = domain.actions_unpruned(state)
        node = self.make_start_node(state, state_t, actions, mask=mask)

        closed = {node: node}
        open = [node]

        num_expanded = 0
        while len(open) > 0:
            if (
                (exp_budget > 0 and num_expanded >= exp_budget)
                or time_budget > 0
                and timer() - start_time >= time_budget
            ):
                return num_expanded, 0, None

            node = heapq.heappop(open)
            num_expanded += 1

            masks = []
            children_to_be_evaluated = []
            state_t_of_children_to_be_evaluated = []
            for a in node.actions:
                new_state = domain.result(node.state, a)
                new_state_actions, mask = domain.actions(a, new_state)

                new_node = self.make_partial_child_node(
                    node, a, new_state_actions, mask, new_state,
                )

                if new_node not in closed:
                    if domain.is_goal(new_state):
                        traj = Trajectory.from_goal_node(
                            domain=domain,
                            goal_node=new_node,
                            num_expanded=num_expanded,
                        )
                        traj = (traj, None)
                        return num_expanded, 0, traj

                    closed[new_node] = new_node
                    if new_state_actions:
                        heapq.heappush(open, new_node)
                        children_to_be_evaluated.append(new_node)
                        state_t = domain.state_tensor(new_state)
                        state_t_of_children_to_be_evaluated.append(state_t)
                        masks.append(mask)

            if len(children_to_be_evaluated) > 0:
                self.finalize_children_nodes(
                    TwoDir.FORWARD,
                    children_to_be_evaluated,
                    state_t_of_children_to_be_evaluated,
                    masks,
                    None,
                )

        print(f"Emptied open list for problem {problem_id}")
        return num_expanded, 0, None
