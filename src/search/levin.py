import copy
import heapq
import math
import time

import numpy as np
import torch as to
import torch.nn.functional as F

from models.utils import mixture_uniform
from search.levin_common import LevinNode, levin_cost, levin_cost_pred_h

from .utils import (
    Direction,
    SearchNode,
    Trajectory,
    get_merged_trajectory,
    reverse_trajectory,
)


class Levin:
    bidirectional = False

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
        device = next(model.parameters()).device

        state = problem.state_tensor().to(device)

        action_logits = model(state)
        if isinstance(action_logits, tuple):
            action_logits = action_logits[0]

        node = LevinNode(
            problem,
            g_cost=0,
            log_prob=1.0,
            levin_cost=1,
            log_action_probs=mixture_uniform(action_logits, self.weight_uniform),
            num_expanded_when_generated=0,
        )

        frontier = []
        reached = {}
        heapq.heappush(frontier, node)
        reached[node] = node

        children_to_be_evaluated = []
        state_t_of_children_to_be_evaluated = []

        num_expanded = 0
        num_generated = 0
        while len(frontier) > 0:

            if (
                (budget and num_expanded >= budget)
                or end_time
                and time.time() > end_time
            ):
                return (False, num_expanded, num_generated, None)

            node = heapq.heappop(frontier)
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

                if new_state.is_solution():
                    solution_len = new_node.g_cost
                    trajectory = Trajectory(new_node, num_expanded, device)
                    if learn:
                        return (
                            solution_len,
                            num_expanded,
                            num_generated,
                            (trajectory,),
                        )
                    else:
                        return solution_len, num_expanded, num_generated, trajectory

                children_to_be_evaluated.append(new_node)
                state_t_of_children_to_be_evaluated.append(new_state.state_tensor())

            batch_states = to.stack(state_t_of_children_to_be_evaluated).to(device)
            action_logits = model(batch_states)

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
                        lc = levin_cost_pred_h(child, predicted_h[i])
                    else:
                        lc = levin_cost(child)
                child.log_action_probs = log_action_probs[i]
                child.levin_cost = lc  # type:ignore

                if child not in reached or child.g_cost < reached[child].g_cost:
                    heapq.heappush(frontier, child)
                    reached[child] = child

            children_to_be_evaluated = []
            state_t_of_children_to_be_evaluated = []

        print("Emptied Open List in problem: ", problem_name)
        return False, num_expanded, num_generated, None
