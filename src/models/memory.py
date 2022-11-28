import math
import sys

import numpy as np
from search.node import SearchNode
from typing import Type


class Trajectory:
    def __init__(self, search_node: SearchNode, num_expanded: int):
        """
        Receives a SearchNode representing a solution to the problem.
        Backtracks the path performed by search, collecting state-action pairs along the way.
        The state-action pairs are stored alongside the number of nodes expanded in an object of type Trajectory,
        which is added to the variable memory.
        """
        self.num_expanded = num_expanded
        if hasattr(search_node, "log_prob"):
            self.solution_prob = math.exp(search_node.log_prob)

        action = search_node.action
        node = search_node.parent
        self.states = []
        self.actions = []
        self.cost_to_gos = []
        cost = 1

        while node:
            self.states.append(node.state)
            self.actions.append(action)
            self.cost_to_gos.append(cost)
            action = node.action
            node = node.parent
            cost += 1


def get_merged_trajectory(
    f_start: SearchNode,
    f_common: SearchNode,
    b_start: SearchNode,
    b_common: SearchNode,
    node_type: Type[SearchNode],
    num_expanded: int,
):
    assert f_common.state == b_common.state
    f_node = f_common
    b_node = b_common
    while b_node.parent:
        new_node = node_type(
            state=b_node.state,
            parent=f_node,
            action=b_node.reverse_action,
            g_cost=f_node.g_cost + 1,
        )
        f_node = new_node
        b_node = b_node.parent

    return Trajectory(b_node, num_expanded)


class Memory:
    def __init__(self):
        self._trajectories = []
        self._max_expanded = -sys.maxsize

    def add_trajectory(self, trajectory):
        if trajectory.num_expanded > self._max_expanded:
            self._max_expanded = trajectory.num_expanded

        self._trajectories.append(trajectory)

    def shuffle_trajectories(self):
        self._random_indices = np.random.permutation(len(self._trajectories))

    def next_trajectory(self):

        for i in range(len(self._trajectories)):
            traj = np.array(self._trajectories)[self._random_indices[i]]
            traj.num_expanded /= self._max_expanded
            yield traj

    def number_trajectories(self):
        return len(self._trajectories)

    def merge_trajectories(self, other):
        for t in other._trajectories:
            self._trajectories.append(t)

    def clear(self):
        self._trajectories.clear()
        self._max_expanded = -sys.maxsize

    def __len__(self):
        return len(self._trajectories)
