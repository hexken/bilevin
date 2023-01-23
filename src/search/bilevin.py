from __future__ import annotations
import time
from typing import TYPE_CHECKING

import torch as to
from torch.nn.functional import log_softmax

from enums import TwoDir
from search.agent import Agent
from search.levin import LevinNode, PriorityQueue, levin_cost, swap_node_contents

if TYPE_CHECKING:
    from domains.domain import Domain, Problem


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
        problem: Problem,
        model,
        budget,
        update_levin_costs=False,
        train=False,
        end_time=None,
    ):
        """ """
        id, f_problem = problem
        b_problem = f_problem.backward_problem()

        f_state = f_problem.reset()
        f_state_t = f_problem.state_tensor(f_state).unsqueeze(0)

        b_state = b_problem.reset()
        b_state_t = b_problem.state_tensor(b_state).unsqueeze(0)

        forward_model, backward_model = model

        f_action_logits = forward_model(f_state_t)
        b_action_logits = backward_model(b_state_t)

        f_log_action_probs = log_softmax(f_action_logits[0], dim=-1)
        b_log_action_probs = log_softmax(b_action_logits[0], dim=-1)

        f_start_node = LevinNode(
            f_state,
            g_cost=0,
            log_prob=1.0,
            levin_cost=1,
            log_action_probs=f_log_action_probs,
        )

        b_start_node = LevinNode(
            b_state,
            g_cost=0,
            log_prob=1.0,
            levin_cost=1,
            log_action_probs=b_log_action_probs,
        )

        f_frontier = PriorityQueue()
        b_frontier = PriorityQueue()
        f_reached = {}
        b_reached = {}
        f_frontier.enqueue(f_start_node)
        b_frontier.enqueue(b_start_node)
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

            b = b_frontier.top()
            f = f_frontier.top()

            if (b and f and f < b) or not b:
                direction = TwoDir.FORWARD
                _problem = f_problem
                _model = forward_model
                _frontier = f_frontier
                _reached = f_reached
                other_problem = b_problem
            else:
                direction = TwoDir.BACKWARD
                _problem = b_problem
                _model = backward_model
                _frontier = b_frontier
                _reached = b_reached
                other_problem = f_problem

            node = _frontier.dequeue()
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
                )
                new_node.levin_cost = levin_cost(new_node)
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
                    _frontier.enqueue(new_node)
                    _problem.update(new_node)

                    children_to_be_evaluated.append(new_node)
                    state_t = _problem.state_tensor(new_state)
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
                action_logits = _model(batch_states)
                log_action_probs = log_softmax(action_logits, dim=-1)

                for i, child in enumerate(children_to_be_evaluated):
                    child.log_action_probs = log_action_probs[i]

                children_to_be_evaluated = []
                state_t_of_children_to_be_evaluated = []

        print(f"Emptied frontiers for problem {id}")
        return False, num_expanded, num_generated, None
