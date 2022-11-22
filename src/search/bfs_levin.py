import copy
import heapq
import math
import time

import numpy as np
import torch as to

from models.memory import Trajectory
from search.node import SearchNode


class LevinNode(SearchNode):
    def __init__(
        self, state, parent, action, g_cost, log_prob, levin_cost, log_action_dist=None
    ):
        super().__init__(state, parent, action, g_cost)
        self.log_prob = log_prob
        self.levin_cost = levin_cost
        self.log_action_dist = log_action_dist

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
        budget,
        start_overall_time,
        time_limit,
        slack_time,
        model,
    ):
        """
        Performs Best-First LTS .

        Returns solution cost, number of nodes expanded, and generated
        """
        if slack_time == 0:
            start_overall_time = time.time()
        start_time = time.time()

        _open = []
        _closed = set()

        num_expanded = 0
        num_generated = 0

        if self._use_learned_heuristic:
            _, action_dist, _ = model(
                np.array([initial_state.get_image_representation()])
            )
        else:
            _, action_dist = model(np.array([initial_state.get_image_representation()]))

        action_distribution_log = np.log(
            (1 - self._mix_epsilon) * action_dist
            + (self._mix_epsilon * (1 / action_dist.shape[1]))
        )

        node = LevinNode(initial_state, None, 0, 0, 0, -1)

        node.set_probability_distribution_actions(action_distribution_log[0])

        heapq.heappush(_open, node)
        _closed.add(initial_state)

        # this array should big enough to have more entries than self._k + the largest number of octions
        predicted_h = np.zeros(10 * self._k)

        children_to_be_evaluated = []
        x_input_of_children_to_be_evaluated = []

        while len(_open) > 0:

            node = heapq.heappop(_open)

            num_expanded += 1

            end_time = time.time()
            if (
                budget > 0 and num_expanded > budget
            ) or end_time - start_overall_time + slack_time > time_limit:
                return -1, num_expanded, num_generated, end_time - start_time

            actions = node.get_game_state().successors_parent_pruning(node.get_action())
            probability_distribution = node.get_probability_distribution_actions()

            for a in actions:
                child = copy.deepcopy(node.get_game_state())
                child.apply_action(a)

                if child.is_solution():
                    end_time = time.time()
                    return (
                        node.get_g() + 1,
                        num_expanded,
                        num_generated,
                        end_time - start_time,
                    )

                child_node = LevinNode(
                    node,
                    child,
                    node.get_p() + probability_distribution[a],
                    node.get_g() + 1,
                    -1,
                    a,
                )

                children_to_be_evaluated.append(child_node)
                x_input_of_children_to_be_evaluated.append(
                    child.get_image_representation()
                )

            if len(children_to_be_evaluated) >= self._k or len(_open) == 0:
                if self._use_learned_heuristic:
                    _, action_dist, predicted_h = model(
                        np.array(x_input_of_children_to_be_evaluated)
                    )
                else:
                    _, action_dist = model(
                        np.array(x_input_of_children_to_be_evaluated)
                    )

                action_distribution_log = np.log(
                    (1 - self._mix_epsilon) * action_dist
                    + (self._mix_epsilon * (1 / action_dist.shape[1]))
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

                    children_to_be_evaluated[i].set_probability_distribution_actions(
                        action_distribution_log[i]
                    )
                    children_to_be_evaluated[i].set_levin_cost(levin_cost)

                    if children_to_be_evaluated[i].get_game_state() not in _closed:
                        heapq.heappush(_open, children_to_be_evaluated[i])
                        _closed.add(children_to_be_evaluated[i].get_game_state())

                children_to_be_evaluated.clear()
                x_input_of_children_to_be_evaluated.clear()
        print("Emptied Open List during search: ", problem_name)
        end_time = time.time()
        return -1, num_expanded, num_generated, end_time - start_time

    def search_for_learning(self, initial_state, problem_name, budget, model):
        """
        Performs Best-First LTS bounded by a search budget.

        Returns Boolean indicating whether the solution was found,
        number of nodes expanded, and number of nodes generated
        """
        _open = []
        _closed = set()

        num_expanded = 0
        num_generated = 0

        #         print('Attempting puzzle ', puzzle_name, ' with budget: ', budget)
        state_t = to.tensor(initial_state.get_image_representation())
        action_dist = model(state_t)[1]

        log_action_dist = np.log(
            (1 - self._mix_epsilon) * action_dist
            + (self._mix_epsilon * (1 / action_dist.shape[0]))
        )

        node = LevinNode(initial_state, None, 0, 0, 0, -1, log_action_dist)

        heapq.heappush(_open, node)
        _closed.add(initial_state)

        # this array should big enough to have more entries than self._k + the largest number of actions
        predicted_h = np.zeros(10 * self._k)

        children_to_be_evaluated = []
        x_input_of_children_to_be_evaluated = []

        while len(_open) > 0:

            node = heapq.heappop(_open)

            num_expanded += 1

            actions = node.state.successors_parent_pruning(node.action)
            log_action_dist = node.log_action_dist

            if num_expanded >= budget:
                return False, None, num_expanded, num_generated

            for a in actions:
                new_state = copy.deepcopy(node.state)
                #                 child = node.get_game_state().copy()
                new_state.apply_action(a)

                new_node = LevinNode(
                    new_state,
                    node,
                    a,
                    node.g_cost + 1,
                    node.log_prob + log_action_dist[a],
                    -1,
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

                batch_states = to.tensor(x_input_of_children_to_be_evaluated)
                if self._use_learned_heuristic:
                    _, _, action_dist, predicted_h = model(to.tensor(batch_states))
                else:
                    _, _, action_dist = model(to.tensor(batch_states))

                log_action_dist = np.log(
                    (1 - self._mix_epsilon) * action_dist
                    + (self._mix_epsilon * (1 / action_dist.shape[0]))
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
                    children_to_be_evaluated[i].log_action_dist = log_action_dist[i]
                    children_to_be_evaluated[i].levin_cost = levin_cost

                    if children_to_be_evaluated[i].state not in _closed:
                        heapq.heappush(_open, children_to_be_evaluated[i])
                        _closed.add(children_to_be_evaluated[i].state)

                children_to_be_evaluated.clear()
                x_input_of_children_to_be_evaluated.clear()
        print("Emptied Open List in problem: ", problem_name)
        return False, None, num_expanded, num_generated
