from __future__ import annotations
import time
from typing import TYPE_CHECKING

import torch as to
from torch.nn.functional import log_softmax

from enums import TwoDir
from search.agent import Agent
from search.levin import PriorityQueue
from search.utils import SearchNode

if TYPE_CHECKING:
    from domains.domain import Domain, Problem


class BiBS(Agent):
    @property
    def bidirectional(cls):
        return True

    @property
    def trainable(cls):
        return False

    def search(
        self,
        problem: Problem,
        model,
        budget,
        update_levin_costs,
        end_time=None,
    ):
        """ """
        problem_id, f_domain = problem
        b_domain = f_domain.backward_domain()

        f_state = f_domain.reset()

        b_state = b_domain.reset()

        f_start_node = SearchNode(
            f_state,
            g_cost=0,
        )

        b_start_node = SearchNode(
            b_state,
            g_cost=0,
        )

        f_frontier = PriorityQueue()
        b_frontier = PriorityQueue()
        f_reached = {}
        b_reached = {}
        f_frontier.enqueue(f_start_node)
        b_frontier.enqueue(b_start_node)
        f_reached[f_start_node] = f_start_node
        b_reached[b_start_node] = b_start_node

        f_domain.update(f_start_node)
        b_domain.update(b_start_node)

        children_to_be_evaluated = []
        state_t_of_children_to_be_evaluated = []

        num_expanded = 0
        num_generated = 0
        while len(f_frontier) > 0 or len(b_frontier) > 0:
            if (
                (budget and num_expanded >= budget)
                or end_time
                and time.time() > end_time
            ):
                return (False, num_expanded, num_generated, None)

            b = b_frontier.top()
            f = f_frontier.top()

            if (b and f and f < b) or not b:
                direction = TwoDir.FORWARD
                _domain = f_domain
                _frontier = f_frontier
                _reached = f_reached
                other_domain = b_domain
            else:
                direction = TwoDir.BACKWARD
                _domain = b_domain
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

                new_node = SearchNode(
                    new_state,
                    g_cost=node.g_cost + 1,
                    parent=node,
                    parent_action=a,
                )
                num_generated += 1

                if new_node not in _reached:
                    trajs = _domain.try_make_solution(
                        new_node, other_domain, num_expanded
                    )

                    if trajs:  # solution found
                        solution_len = len(trajs[0])
                        assert solution_len == len(trajs[1])
                        return solution_len, num_expanded, num_generated, trajs[0]

                    _reached[new_node] = new_node
                    _frontier.enqueue(new_node)
                    _domain.update(new_node)

        print(f"Emptied frontiers for problem {problem_id}")
        return False, num_expanded, num_generated, None
