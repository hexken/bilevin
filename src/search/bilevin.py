import copy
import heapq
import math
import time

import numpy as np
import torch as to
import torch.nn.functional as F

from models.utils import mixture_uniform
from search.levin_common import LevinNode, levin_cost

from .utils import (
    Direction,
    SearchNode,
    Trajectory,
    get_merged_trajectory,
    reverse_trajectory,
)


class BiLevin:
    bidirectional = True

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
        b_problem = problem.get_backward_problem()

        f_state = problem.state_tensor().to(device)
        b_state = b_problem.state_tensor().to(device)
        initial_state = f_state.clone()
        dims = [1] * initial_state.dim()
        initial_state_repeated = initial_state.repeat(problem.num_actions, *dims)

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
            log_action_probs=mixture_uniform(f_action_logits, self.weight_uniform),
            num_expanded_when_generated=0,
        )

        b_start_node = LevinNode(
            b_problem,
            g_cost=0,
            log_prob=1.0,
            levin_cost=1,
            log_action_probs=mixture_uniform(b_action_logits, self.weight_uniform),
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

            if b_frontier[0] < f_frontier[0]:
                direction = Direction.BACKWARD
                _model = backward_model
                _frontier = b_frontier
                _reached = b_reached
                _other_reached = f_reached
            else:
                direction = Direction.FORWARD
                _model = forward_model
                _frontier = f_frontier
                _reached = f_reached
                _other_reached = b_reached

            node = heapq.heappop(_frontier)
            num_expanded += 1
            actions = node.state.successors_parent_pruning(node.action)
            for a in actions:
                # todo vectorize this? Will depend on how I re-implement envs
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
                num_generated += 1

                children_to_be_evaluated.append(new_node)
                state_t_of_children_to_be_evaluated.append(new_state.state_tensor())

            batch_states = to.stack(state_t_of_children_to_be_evaluated).to(device)
            if direction == Direction.BACKWARD:
                action_logits = _model(
                    batch_states, initial_state_repeated[: len(batch_states)]
                )
            else:
                action_logits = _model(batch_states)

            predicted_h = None
            if isinstance(action_logits, tuple):
                action_logits, predicted_h = action_logits

            log_action_probs = mixture_uniform(action_logits, self.weight_uniform)

            for i, child in enumerate(children_to_be_evaluated):
                # todo vectorize this loop!

                if self.estimated_probability_to_go:
                    pass
                    # levin_cost = self.get_levin_cost_star(
                    #     children_to_be_evaluated[i], predicted_h[i]
                    # )
                else:
                    if predicted_h:
                        lc = levin_cost(child, predicted_h[i])
                    else:
                        lc = levin_cost(child, None)
                child.log_action_probs = log_action_probs[i]
                child.levin_cost = lc  # type:ignore

                if child not in _reached or child.g_cost < _reached[child].g_cost:
                    heapq.heappush(_frontier, child)
                    _reached[child] = child
                    if child in _other_reached:  # solution found
                        f_common_node = f_reached[child]
                        b_common_node = b_reached[child]

                        forward_traj = get_merged_trajectory(
                            f_common_node,
                            b_common_node,
                            LevinNode,
                            num_expanded,
                        )

                        if learn:
                            backward_traj = get_merged_trajectory(
                                b_common_node,
                                f_common_node,
                                LevinNode,
                                num_expanded,
                            )
                            return (
                                len(forward_traj),
                                num_expanded,
                                num_generated,
                                (forward_traj, backward_traj),
                            )
                        else:
                            return (
                                len(forward_traj),
                                num_expanded,
                                num_generated,
                                forward_traj,
                            )

            children_to_be_evaluated = []
            state_t_of_children_to_be_evaluated = []

        print("Emptied Open List in problem: ", problem_name)
        return False, num_expanded, num_generated, None
