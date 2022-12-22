import copy
import heapq
import math
import time

import numpy as np
import torch as to
import torch.nn.functional as F

from models.utils import mixture_uniform
from search.agent import Agent
from search.utils import SearchNode, Trajectory


class Levin(Agent):
    @property
    def bidirectional(cls):
        return False

    def __init__(
        self,
        weight_uniform: float = 0.0,
    ):
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

        state = problem.state_tensor(device)

        action_logits = model(state)

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
            if not actions:
                continue

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
            log_action_probs = mixture_uniform(action_logits, self.weight_uniform)

            for i, child in enumerate(children_to_be_evaluated):
                # todo vectorize?
                lc = levin_cost(child)
                child.log_action_probs = log_action_probs[i]
                child.levin_cost = lc  # type:ignore

                if child not in reached:  # or child.g_cost < reached[child].g_cost:
                    heapq.heappush(frontier, child)
                    reached[child] = child

            children_to_be_evaluated = []
            state_t_of_children_to_be_evaluated = []

        print("Emptied Open List in problem: ", problem_name)
        return False, num_expanded, num_generated, None


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


def levin_cost(node: LevinNode):
    return math.log(node.g_cost + 1) - node.log_prob
