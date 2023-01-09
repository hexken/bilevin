from __future__ import annotations
import heapq
import math
import time
from typing import Optional, TYPE_CHECKING

import torch as to

from models.utils import mixture_uniform
from search.agent import Agent
from search.utils import SearchNode, Trajectory

if TYPE_CHECKING:
    from domains.domain import State


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
        model,
        budget,
        train=False,
        end_time=None,
    ):
        """ """
        device = next(model.parameters()).device

        state = problem.reset()
        state_t = problem.state_tensor(state, device).unsqueeze(0)

        action_logits = model(state_t)

        node = LevinNode(
            state,
            g_cost=0,
            log_prob=1.0,
            levin_cost=1,
            log_action_probs=mixture_uniform(action_logits[0], self.weight_uniform),
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
                return False, num_expanded, num_generated, None

            node = heapq.heappop(frontier)
            num_expanded += 1

            actions = problem.actions(node.parent_action, node.state)
            if not actions:
                continue

            for a in actions:
                new_state = problem.result(node.state, a)

                new_node = LevinNode(
                    new_state,
                    node,
                    a,
                    node.g_cost + 1,
                    node.log_prob + node.log_action_probs[a].item(),
                    num_expanded_when_generated=num_expanded,
                )
                num_generated += 1

                if new_node not in reached:
                    if problem.is_goal(new_state):
                        solution_len = new_node.g_cost
                        traj = Trajectory(problem, new_node, num_expanded, device)
                        if train:
                            traj = (traj,)
                        return solution_len, num_expanded, num_generated, traj

                    reached[new_node] = new_node
                    children_to_be_evaluated.append(new_node)

                state_t = problem.state_tensor(new_state, device)
                state_t_of_children_to_be_evaluated.append(state_t)

            batch_states = to.stack(state_t_of_children_to_be_evaluated)
            action_logits = model(batch_states)
            log_action_probs = mixture_uniform(action_logits, self.weight_uniform)

            for i, child in enumerate(children_to_be_evaluated):
                lc = levin_cost(child)
                child.log_action_probs = log_action_probs[i]
                child.levin_cost = lc
                heapq.heappush(frontier, child)

            children_to_be_evaluated = []
            state_t_of_children_to_be_evaluated = []

        # todo log empty frontier?
        return False, num_expanded, num_generated, None


class LevinNode(SearchNode):
    def __init__(
        self,
        state: Optional[State],
        parent: Optional[SearchNode] = None,
        parent_action=None,
        g_cost: Optional[float] = None,
        log_prob: Optional[float] = None,
        levin_cost: Optional[float] = None,
        log_action_probs: Optional[to.Tensor] = None,
        num_expanded_when_generated: Optional[int] = None,
    ):
        super().__init__(state, parent, parent_action, g_cost)
        self.log_prob = log_prob
        self.levin_cost = levin_cost
        self.log_action_probs = log_action_probs

    def __lt__(self, other):
        """
        used by the heap
        """
        return self.levin_cost < other.levin_cost


def levin_cost(node: LevinNode):
    return math.log(node.g_cost + 1) - node.log_prob  # type:ignore
