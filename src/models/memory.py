import math
import sys

import numpy as np


class Trajectory:
    def __init__(self, search_node, num_expanded):
        """
        Receives a SearchNode representing a solution to the problem.
        Backtracks the path performed by search, collecting state-action pairs along the way.
        The state-action pairs are stored alongside the number of nodes expanded in an object of type Trajectory,
        which is added to the variable memory.
        """
        self.num_expanded = num_expanded
        self.solution_prob = math.exp(search_node.log_prob)

        node = search_node.parent
        self.states = []
        self.actions = []
        self.cost_to_gos = []
        cost = 1

        while node:
            self.states.append(node.state)
            self.actions.append(node.action)
            self.cost_to_gos.append(cost)
            node = node.parent
            cost += 1


class Memory:
    def __init__(self):
        self._trajectories = []
        self._max_expanded = -sys.maxsize

    def add_trajectory(self, trajectory):
        if trajectory.get_expanded() > self._max_expanded:
            self._max_expanded = trajectory.get_expanded()

        self._trajectories.append(trajectory)

    def shuffle_trajectories(self):
        self._random_indices = np.random.permutation(len(self._trajectories))

    def next_trajectory(self):

        for i in range(len(self._trajectories)):
            traject = np.array(self._trajectories)[self._random_indices[i]]
            traject.normalize_expanded(self._max_expanded)
            yield traject

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
