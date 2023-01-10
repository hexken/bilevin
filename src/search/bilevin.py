from __future__ import annotations
import heapq
import time
from typing import TYPE_CHECKING

import torch as to

from enums import TwoDir
from models.utils import mixture_uniform
from search.agent import Agent
from search.levin import LevinNode, levin_cost

if TYPE_CHECKING:
    from domains.domain import Domain


class BiLevin(Agent):
    @property
    def bidirectional(cls):
        return True

    def __init__(
        self,
        weight_uniform: float = 0.0,
    ):
        self.weight_uniform = weight_uniform

    def search(
        self,
        problem: tuple[int, Domain],
        model,
        budget,
        train=False,
        end_time=None,
    ):
        """ """
        id, domain = problem
        f_problem = domain
        b_problem = domain.backward_problem()

        f_state = domain.reset()
        f_state_t = f_problem.state_tensor(f_state).unsqueeze(0)

        b_state = b_problem.reset()
        b_state_t = b_problem.state_tensor(b_state).unsqueeze(0)

        forward_model, backward_model = model

        f_action_logits = forward_model(f_state_t)
        b_action_logits = backward_model(b_state_t)

        f_start_node = LevinNode(
            f_state,
            g_cost=0,
            log_prob=1.0,
            levin_cost=1,
            log_action_probs=mixture_uniform(f_action_logits[0], self.weight_uniform),
            num_expanded_when_generated=0,
        )

        b_start_node = LevinNode(
            b_state,
            g_cost=0,
            log_prob=1.0,
            levin_cost=1,
            log_action_probs=mixture_uniform(b_action_logits[0], self.weight_uniform),
            num_expanded_when_generated=0,
        )

        f_frontier = []
        b_frontier = []
        f_reached = {}
        b_reached = {}
        heapq.heappush(f_frontier, f_start_node)
        heapq.heappush(b_frontier, b_start_node)
        f_reached[f_start_node] = f_start_node
        b_reached[b_start_node] = b_start_node

        f_problem.update(f_start_node)
        b_problem.update(b_start_node)

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

            if b_frontier[0] < f_frontier[0]:
                _problem = b_problem
                _model = backward_model
                _frontier = b_frontier
                _reached = b_reached
                other_problem = f_problem
            else:
                _problem = f_problem
                _model = forward_model
                _frontier = f_frontier
                _reached = f_reached
                other_problem = b_problem

            node = heapq.heappop(_frontier)
            num_expanded += 1
            actions = _problem.actions(node.parent_action, node.state)
            if not actions:
                continue

            for a in actions:
                new_state = _problem.result(node.state, a)

                new_node = LevinNode(
                    new_state,
                    node,
                    a,
                    node.g_cost + 1,
                    node.log_prob + node.log_action_probs[a].item(),
                    num_expanded_when_generated=num_expanded,
                )
                num_generated += 1

                if new_node not in _reached:
                    trajs = _problem.try_make_solution(
                        new_node, other_problem, num_expanded
                    )

                    if trajs:  # solution found
                        solution_len = len(trajs[0])
                        assert solution_len == len(trajs[1])
                        if not train:
                            trajs = trajs[0]
                        return solution_len, num_expanded, num_generated, trajs

                    _reached[new_node] = new_node
                    _problem.update(new_node)
                    children_to_be_evaluated.append(new_node)

                state_t = _problem.state_tensor(new_state)
                state_t_of_children_to_be_evaluated.append(state_t)

            batch_states = to.stack(state_t_of_children_to_be_evaluated)
            action_logits = _model(batch_states)
            log_action_probs = mixture_uniform(action_logits, self.weight_uniform)

            for i, child in enumerate(children_to_be_evaluated):
                lc = levin_cost(child)
                child.log_action_probs = log_action_probs[i]
                child.levin_cost = lc
                heapq.heappush(_frontier, child)

            children_to_be_evaluated = []
            state_t_of_children_to_be_evaluated = []

        print(f"Emptied frontiers for problem {id}")
        return False, num_expanded, num_generated, None
