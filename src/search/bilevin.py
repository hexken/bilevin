import copy
import heapq
import math
import time

import numpy as np
import torch as to
import torch.nn.functional as F

from utils import Direction

from .utils import (
    SearchNode,
    Trajectory,
    reverse_trajectory,
    get_merged_trajectory,
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
        num_expanded_when_generated=None,
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


class BiLevin:
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
        device = next(model[0].parameters()).device
        f_open = []
        b_open = []
        f_closed = {}
        b_closed = {}

        num_expanded = 0
        num_generated = 0

        b_problem = problem.get_backward_problem()

        f_state = problem.state_tensor().to(device)
        b_state = b_problem.state_tensor().to(device)
        initial_state = f_state.clone()

        forward_model, backward_model = model

        f_action_logits = forward_model(f_state)
        b_action_logits = backward_model(b_state, initial_state)

        if isinstance(f_action_logits, tuple):
            f_action_logits = f_action_logits[0]
            b_action_logits = b_action_logits[0]

        f_start_node = LevinNode(
            problem,
            g_cost=0,
            log_prob=1.0,
            levin_cost=1,
            log_action_probs=self.mixture_uniform(f_action_logits),
            num_expanded_when_generated=0,
        )

        b_start_node = LevinNode(
            b_problem,
            g_cost=0,
            log_prob=1.0,
            levin_cost=1,
            log_action_probs=self.mixture_uniform(b_action_logits),
            num_expanded_when_generated=0,
        )

        heapq.heappush(f_open, f_start_node)
        heapq.heappush(b_open, b_start_node)
        f_closed[problem] = problem
        b_closed[b_problem] = b_problem

        children_to_be_evaluated = []
        state_t_of_children_to_be_evaluated = []

        while len(f_open) > 0 and len(b_open) > 0:

            if (
                (budget and num_expanded >= budget)
                or end_time
                and time.time() > end_time
            ):
                return (False, num_expanded, num_generated, None)

            if f_open[0] < b_open[0]:
                direction = Direction.FORWARD
                _model = forward_model
                _open = f_open
                _closed = f_closed
            else:
                direction = Direction.BACKWARD
                _model = backward_model
                _open = b_open
                _closed = b_closed

            node = heapq.heappop(_open)
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
                    num_expanded_when_generated=num_expanded,
                )

                if new_state.is_solution():
                    solution_len = new_node.g_cost
                    trajectory = Trajectory(new_node, num_expanded)
                    if learn:
                        backward_trajectory = reverse_trajectory(trajectory)
                        return (
                            solution_len,
                            num_expanded,
                            num_generated,
                            (trajectory, backward_trajectory),
                        )
                    else:
                        return solution_len, num_expanded, num_generated, trajectory

                children_to_be_evaluated.append(new_node)
                state_t_of_children_to_be_evaluated.append(new_state.state_tensor())

            batch_states = to.stack(state_t_of_children_to_be_evaluated).to(device)
            if direction == Direction.BACKWARD:
                initial_states = initial_state.repeat(len(batch_states), 1)
                batch_states = (batch_states, initial_states)
                action_logits = _model(batch_states, initial_states)
            else:
                action_logits = _model(batch_states)

            predicted_h = None
            if isinstance(action_logits, tuple):
                action_logits, predicted_h = action_logits

            log_action_probs = self.mixture_uniform(action_logits)

            for i, child in enumerate(children_to_be_evaluated):
                num_generated += 1

                if self.estimated_probability_to_go:
                    pass
                    # levin_cost = self.get_levin_cost_star(
                    #     children_to_be_evaluated[i], predicted_h[i]
                    # )
                else:
                    if predicted_h:
                        levin_cost = self.levin_cost(child, predicted_h[i])
                    else:
                        levin_cost = self.levin_cost(child, None)
                child.log_action_probs = log_action_probs[i]
                child.levin_cost = levin_cost  # type:ignore

                if child.state not in _closed:
                    heapq.heappush(_open, child)
                    _closed[child] = child.state

                children_to_be_evaluated = []
                state_t_of_children_to_be_evaluated = []

            num_expanded += 1

            for key in f_closed:
                if key in b_closed:
                    f_common_node = f_closed[key]
                    b_common_node = b_closed[key]

                    forward_traj = get_merged_trajectory(
                        f_start_node,
                        f_common_node,
                        b_start_node,
                        b_common_node,
                        LevinNode,
                        num_expanded,
                    )
                    if learn:
                        backward_traj = get_merged_trajectory(
                            b_start_node,
                            b_common_node,
                            f_start_node,
                            f_common_node,
                            LevinNode,
                            num_expanded,
                        )
                        return (
                            f_common_node.g_cost + b_common_node.g_cost - 1,
                            num_expanded,
                            num_generated,
                            (forward_traj, backward_traj),
                        )
                    else:
                        return (
                            f_common_node.g_cost + b_common_node.g_cost - 1,
                            num_expanded,
                            num_generated,
                            forward_traj,
                        )

        print("Emptied Open List in problem: ", problem_name)
        return False, num_expanded, num_generated, None
