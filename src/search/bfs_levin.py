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
        used by the heap
        """
        return self.levin_cost < other.levin_cost


class BFSLevin:
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

    def get_levin_cost_star(self, child_node, predicted_h):
        if self.use_learned_heuristic and self.use_default_heuristic:
            max_h = max(predicted_h, child_node.get_game_state().heuristic_value())
            return math.log(max_h + child_node.get_g()) - (
                child_node.get_p() * (1 + (max_h / child_node.get_g()))
            )
        elif self.use_learned_heuristic:
            if predicted_h < 0:
                predicted_h = 0

            return math.log(predicted_h + child_node.get_g()) - (
                child_node.get_p() * (1 + (predicted_h / child_node.get_g()))
            )
        else:
            h_value = child_node.get_game_state().heuristic_value()
            return math.log(h_value + child_node.g_cost) - (
                child_node.get_p() * (1 + (h_value / child_node.get_g()))
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

    def search(
        self,
        initial_state,
        problem_name,
        model,
        budget,
        learn=False,
        end_time=None,
    ):
        """ """
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
            (1 - self.weight_uniform) * action_probs
            + self.weight_uniform * (1 / len(action_probs))
        )

        node = LevinNode(
            initial_state,
            None,
            g_cost=0,
            log_prob=1.0,
            log_action_probs=log_action_probs,
        )

        heapq.heappush(_open, node)
        _closed.add(initial_state)

        children_to_be_evaluated = []
        x_input_of_children_to_be_evaluated = []

        while len(_open) > 0:
            node = heapq.heappop(_open)

            if (
                (budget and num_expanded >= budget)
                or end_time
                and time.time() > end_time
            ):
                return (False, num_expanded, num_generated, None)

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

                if new_state.is_solution():
                    print(
                        "Solved problem: ",
                        problem_name,
                        " expanded ",
                        num_expanded,
                        " with budget: ",
                        budget,
                    )
                    if learn:
                        trajectory = Trajectory(new_node, num_expanded)
                        return True, num_expanded, num_generated, trajectory
                    else:
                        return True, num_expanded, num_generated, None

                children_to_be_evaluated.append(new_node)
                x_input_of_children_to_be_evaluated.append(
                    new_state.get_image_representation()
                )
            num_expanded += 1
            if (
                len(children_to_be_evaluated) >= self.batch_size_expansions
                or len(_open) == 0
            ):

                batch_states = to.tensor(np.array(x_input_of_children_to_be_evaluated))

                action_logits = model(batch_states)
                if isinstance(action_logits, tuple):
                    action_logits, predicted_h = action_logits

                action_probs = to.log(F.log_softmax(action_logits, dim=0))
                log_action_probs = to.log(
                    (1 - self.weight_uniform) * action_probs
                    + self.weight_uniform * (1 / len(action_probs))
                )

                for i in range(len(children_to_be_evaluated)):
                    num_generated += 1

                    if self.estimated_probability_to_go:
                        levin_cost = self.get_levin_cost_star(
                            children_to_be_evaluated[i], predicted_h[i]
                        )
                    else:
                        if i >= len(predicted_h):
                            levin_cost = self.levin_cost(
                                children_to_be_evaluated[i], None
                            )
                        else:
                            levin_cost = self.levin_cost(
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
        return False, num_expanded, num_generated, None
