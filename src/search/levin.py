import copy
import heapq
import math
import time

from enum import Enum
import numpy as np
import torch as to
import torch.nn.functional as F

from .utils import (
    Trajectory,
    SearchNode,
    get_merged_trajectory,
    convert_to_backward_trajectory,
)


class LevinNode(SearchNode):
    def __init__(
        self,
        state,
        parent=None,
        action=None,
        g_cost=None,
        log_prob=None,
        levin_cost=None,
        log_action_probs=None,
    ):
        super().__init__(state, parent, action, g_cost)
        self.log_prob = log_prob
        self.levin_cost = levin_cost
        self.log_action_probs = log_action_probs

    def __lt__(self, other):
        """
        used by the heap
        """
        return self.levin_cost < other.levin_cost


class Direction(Enum):
    FORWARD = 0
    BACKWARD = 1


class Levin:
    def __init__(
        self,
        use_default_heuristic=True,
        use_learned_heuristic=False,
        estimated_probability_to_go=True,
        batch_size_expansions=32,
        weight_uniform=0.0,
    ):
        self.use_default_heuristic = use_default_heuristic
        self.use_learned_heuristic = use_learned_heuristic
        self.estimated_probability_to_go = estimated_probability_to_go
        self.batch_size_expansions = batch_size_expansions
        self.weight_uniform = weight_uniform

    def get_levin_cost_star(self, node, predicted_h):
        if self.use_learned_heuristic and self.use_default_heuristic:
            max_h = max(predicted_h, node.state.heuristic_value())
            return math.log(max_h + node.g_cost) - (
                node.log_prob * (1 + (max_h / node.g_cost))
            )
        elif self.use_learned_heuristic:
            if predicted_h < 0:
                predicted_h = 0

            return math.log(predicted_h + node.get_g()) - (
                node.get_p() * (1 + (predicted_h / node.get_g()))
            )
        else:
            h_value = node.get_game_state().heuristic_value()
            return math.log(h_value + node.g_cost) - (
                node.get_p() * (1 + (h_value / node.get_g()))
            )

    def levin_cost(self, node, predicted_h):
        # todo these costs don't look right
        if self.use_learned_heuristic and self.use_default_heuristic:
            max_h = max(predicted_h, node.state.heuristic_value())
            return math.log(max_h + node.g_cost) - node.log_prob
        elif self.use_learned_heuristic:
            if predicted_h < 0:
                predicted_h = 0
            return math.log(predicted_h + node.g_cost) - node.log_prob
        elif self.use_default_heuristic:
            return math.log(node.state.heuristic_value() + node.g_cost) - node.log_prob
        return math.log(node.g_cost) - node.log_prob

    def mixture_uniform(self, logits):
        probs = to.exp(F.log_softmax(logits, dim=0))
        log_probs = to.log(
            (1 - self.weight_uniform) * probs + self.weight_uniform * (1 / len(probs))
        )
        return log_probs

    def search(
        self,
        problem,
        problem_name,
        model,
        budget,
        learn=False,
        end_time=None,
    ):
        """ """
        f_open = []
        b_open = []
        f_closed = set()
        b_closed = set()

        num_expanded = 0
        num_generated = 0

        b_problem = problem.get_backward_problem()

        f_state = problem.state_tensor().to(model.device)
        b_state = b_problem.state_tensor().to(model.device)

        action_logits = model(to.stack((f_state, b_state)))

        if isinstance(action_logits, tuple):
            action_logits = action_logits[0]

        log_action_probs = self.mixture_uniform(action_logits)

        f_node = LevinNode(
            problem,
            None,
            g_cost=0,
            log_prob=1.0,
            log_action_probs=log_action_probs[0],
        )

        b_node = LevinNode(
            b_problem,
            None,
            g_cost=0,
            log_prob=1.0,
            log_action_probs=log_action_probs[1],
        )

        heapq.heappush(f_open, f_node)
        heapq.heappush(b_open, b_node)
        f_closed.add(problem)
        b_closed.add(b_problem)

        children_to_be_evaluated = []
        x_input_of_children_to_be_evaluated = []

        while len(f_open) > 0 and len(b_open) > 0:

            if (
                (budget and num_expanded >= budget)
                or end_time
                and time.time() > end_time
            ):
                return (False, num_expanded, num_generated, None)

            if f_open[0] < b_open[0]:
                node = heapq.heappop(f_open)
                direction = Direction.FORWARD
                _open = f_open
                _closed = f_closed
            else:
                node = heapq.heappop(b_open)
                direction = Direction.BACKWARD
                _open = b_open
                _closed = b_closed

            actions = node.state.successors_parent_pruning(node.action)
            for a in actions:
                new_state = copy.deepcopy(node.state)
                new_state.apply_action(a)

                new_node = LevinNode(
                    new_state,
                    node,
                    a,
                    node.g_cost + 1,
                    node.log_prob + node.log_action_probs[a],
                )

                # todo or frontiers meet..
                if new_state.is_solution():
                    solution_len = new_node.g_cost
                    if learn:
                        trajectory = models.memory.Trajectory(new_node, num_expanded)
                        return solution_len, num_expanded, num_generated, trajectory
                    else:
                        return solution_len, num_expanded, num_generated, None

                children_to_be_evaluated.append(new_node)
                x_input_of_children_to_be_evaluated.append(new_state.state_tensor())

            batch_states = to.stack(x_input_of_children_to_be_evaluated).to(
                model.device
            )
            action_logits = model(batch_states)
            predicted_h = None
            if isinstance(action_logits, tuple):
                action_logits, predicted_h = action_logits

            log_action_probs = self.mixture_uniform(action_logits)

            for i in range(len(children_to_be_evaluated)):
                num_generated += 1

                if self.estimated_probability_to_go:
                    levin_cost = self.get_levin_cost_star(
                        children_to_be_evaluated[i], predicted_h[i]
                    )
                else:
                    if predicted_h:
                        levin_cost = self.levin_cost(
                            children_to_be_evaluated[i], predicted_h[i]
                        )
                    else:
                        levin_cost = self.levin_cost(children_to_be_evaluated[i], None)
                children_to_be_evaluated[i].log_action_probs = log_action_probs[i]
                children_to_be_evaluated[i].levin_cost = levin_cost

                if children_to_be_evaluated[i].state not in _closed:
                    heapq.heappush(_open, children_to_be_evaluated[i])
                    _closed.add(children_to_be_evaluated[i].state)

                children_to_be_evaluated = []
                x_input_of_children_to_be_evaluated = []

            num_expanded += 1

        print("Emptied Open List in problem: ", problem_name)
        return False, num_expanded, num_generated, None
