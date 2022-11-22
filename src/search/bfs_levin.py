import copy
import heapq
import math
import time

import numpy as np
import torch as to
import torch.nn.functional as F

from models.memory import Trajectory
from search.node import SearchNode


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
        Function less-than used by the heap
        """
        return self.levin_cost < other.levin_cost


class BFSLevin:
    def __init__(
        self,
        use_heuristic=True,
        use_learned_heuristic=False,
        estimated_probability_to_go=True,
        k_expansions=32,
        mix_epsilon=0.0,
    ):
        self._use_heuristic = use_heuristic
        self._use_learned_heuristic = use_learned_heuristic
        self._estimated_probability_to_go = estimated_probability_to_go
        self._k = k_expansions
        self._mix_epsilon = mix_epsilon

    def get_levin_cost_star(self, child_node, predicted_h):
        if self._use_learned_heuristic and self._use_heuristic:
            max_h = max(predicted_h, child_node.get_game_state().heuristic_value())
            return math.log(max_h + child_node.get_g()) - (
                child_node.get_p() * (1 + (max_h / child_node.get_g()))
            )
        elif self._use_learned_heuristic:
            if predicted_h < 0:
                predicted_h = 0

            return math.log(predicted_h + child_node.get_g()) - (
                child_node.get_p() * (1 + (predicted_h / child_node.get_g()))
            )
        else:
            h_value = child_node.get_game_state().heuristic_value()
            return math.log(h_value + child_node.get_g()) - (
                child_node.get_p() * (1 + (h_value / child_node.get_g()))
            )

    def get_levin_cost(self, child_node, predicted_h):
        # todo these costs don't look right
        if self._use_learned_heuristic and self._use_heuristic:
            max_h = max(predicted_h, child_node.get_game_state().heuristic_value())
            return math.log(max_h + child_node.g_cost) - child_node.log_prob
        elif self._use_learned_heuristic:
            if predicted_h < 0:
                predicted_h = 0
            return math.log(predicted_h + child_node.g_cost) - child_node.log_prob
        elif self._use_heuristic:
            return (
                math.log(child_node.state.heuristic_value() + child_node.g_cost)
                - child_node.log_prob
            )
        return math.log(child_node.g_cost) - child_node.log_prob

    def search(
        self,
        initial_state,
        problem_name,
        model,
        budget,
        learn=False,
        limit_time=False,
        start_time=None,
        time_limit=None,
        slack_time=0,
    ):
        """
        Performs Best-First LTS bounded by a search budget.

        Returns Boolean indicating whether the solution was found,
        number of nodes expanded, and number of nodes generated
        """
        if limit_time:
            assert start_time is not None
            assert time_limit is not None
        else:
            raise ValueError("Invalid time limits specified")

        _open = []
        _closed = set()

        num_expanded = 0
        num_generated = 0

        state_t = to.tensor(initial_state.get_image_representation())
        acion_logits = model(state_t)
        if isinstance(acion_logits, tuple):
            acion_logits = acion_logits[0]

        action_probs = to.log(F.log_softmax(acion_logits, dim=0))
        log_action_probs = to.log(
            (1 - self._mix_epsilon) * action_probs
            + self._mix_epsilon * (1 / len(action_probs))
        )

        node = LevinNode(initial_state, None, log_action_probs=log_action_probs)

        heapq.heappush(_open, node)
        _closed.add(initial_state)

        # this array should big enough to have more entries than self._k + the largest number of actions
        # todo what is this even for?
        predicted_h = np.zeros(10 * self._k)

        children_to_be_evaluated = []
        x_input_of_children_to_be_evaluated = []

        while len(_open) > 0:
            node = heapq.heappop(_open)
            num_expanded += 1

            end_time = time.time()
            if (
                budget and num_expanded > budget
            ) or limit_time and end_time - start_time + slack_time > time_limit:
                return False, num_expanded, num_generated, end_time - start_time, puzzle_name
            actions = node.state.successors_parent_pruning(node.action)

            if num_expanded >= budget:
                return False, None, num_expanded, num_generated

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

                if new_state.is_solution():
                    print(
                        "Solved problem: ",
                        problem_name,
                        " expanded ",
                        num_expanded,
                        " with budget: ",
                        budget,
                    )
                    trajectory = Trajectory(new_node, num_expanded)
                    return True, trajectory, num_expanded, num_generated

                children_to_be_evaluated.append(new_node)
                x_input_of_children_to_be_evaluated.append(
                    new_state.get_image_representation()
                )

            if len(children_to_be_evaluated) >= self._k or len(_open) == 0:
                # todo work with log_probs everywhere
                # use ai book api
                # check how probs are computed

                batch_states = to.tensor(x_input_of_children_to_be_evaluated)
                if self._use_learned_heuristic:
                    action_logits, predicted_h = model(to.tensor(batch_states))
                else:
                    action_logits, _ = model(to.tensor(batch_states))

                acion_logits = model(state_t)
                if isinstance(acion_logits, tuple):
                    acion_logits = acion_logits[0]

                action_probs = to.log(F.log_softmax(acion_logits, dim=0))
                log_action_probs = to.log(
                    (1 - self._mix_epsilon) * action_probs
                    + self._mix_epsilon * (1 / len(action_probs))
                )

                for i in range(len(children_to_be_evaluated)):
                    num_generated += 1

                    if self._estimated_probability_to_go:
                        levin_cost = self.get_levin_cost_star(
                            children_to_be_evaluated[i], predicted_h[i]
                        )
                    else:
                        if i >= len(predicted_h):
                            levin_cost = self.get_levin_cost(
                                children_to_be_evaluated[i], None
                            )
                        else:
                            levin_cost = self.get_levin_cost(
                                children_to_be_evaluated[i], predicted_h[i]
                            )
                    children_to_be_evaluated[i].log_action_probs = log_action_probs[i]
                    children_to_be_evaluated[i].levin_cost = levin_cost

                    if children_to_be_evaluated[i].state not in _closed:
                        heapq.heappush(_open, children_to_be_evaluated[i])
                        _closed.add(children_to_be_evaluated[i].state)

                children_to_be_evaluated.clear()
                x_input_of_children_to_be_evaluated.clear()
        print("Emptied Open List in problem: ", problem_name)
        return False, None, num_expanded, num_generated
