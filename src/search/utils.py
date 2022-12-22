import math
from math import exp
from typing import Type

import numpy as np
import torch as to


class SearchNode:
    def __init__(self, state, parent, action, g_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.g_cost = g_cost
        self.reverse_action = state.reverse_action(action) if action and state else None

    def __eq__(self, other):
        """
        Verify if two SearchNodes are identical by verifying the
         state in the nodes.
        """
        return self.state == other.state

    def __lt__(self, other):
        """
        less-than used by the heap
        """
        return self.g_cost < other.g_cost

    def __hash__(self):
        """
        Hash function used in the closed list
        """
        return self.state.__hash__()


class Trajectory:
    def __init__(self, final_node: SearchNode, num_expanded: int, device: to.device):
        """
        Receives a SearchNode representing a solution to the problem.
        Backtracks the path performed by search, collecting state-action pairs along the way.
        """
        self.num_expanded = num_expanded
        if hasattr(final_node, "log_prob"):
            self.solution_prob = exp(final_node.log_prob)  # type:ignore

        action = final_node.action
        node = final_node.parent
        self.goal = final_node.state.state_tensor(device)
        states = []
        actions = []
        cost_to_gos = []
        cost = 1

        while node:
            states.append(node.state.state_tensor(device))
            actions.append(action)
            cost_to_gos.append(cost)
            action = node.action
            node = node.parent
            cost += 1

        self.states = to.stack(states[::-1])
        self.actions = to.tensor(actions[::-1], device=device)
        self.cost_to_gos = to.tensor(cost_to_gos[::-1], device=device)

    def to(self, device):
        self.states = self.states.to(device)
        self.actions = self.actions.to(device)
        self.cost_to_gos = self.cost_to_gos.to(device)
        self.device = device

    def __len__(self):
        return len(self.states)


def reverse_trajectory(f_trajectory: Trajectory, device=None):
    """
    Returns a new a trajectory that is the reverse of f_trajectory.
    """
    device = device if device else f_trajectory.device
    dummy_node = SearchNode(state=None, parent=None, action=None, g_cost=None)
    b_trajectory = Trajectory(
        dummy_node, num_expanded=f_trajectory.num_expanded, device=device
    )
    if hasattr(f_trajectory, "solution_prob"):
        b_trajectory.solution_prob = f_trajectory.solution_prob
    b_trajectory.goal = f_trajectory.goal
    b_trajectory.states = to.flip(f_trajectory.states, dims=[0])
    b_trajectory.actions = to.flip(f_trajectory.actions, dims=[0])
    b_trajectory.cost_to_gos = to.flip(f_trajectory.cost_to_gos, dims=[0])
    return b_trajectory


def get_merged_trajectory(
    f_common: SearchNode,
    b_common: SearchNode,
    node_type: Type[SearchNode],
    num_expanded: int,
    device=to.device("cpu"),
):
    """
    Returns a new trajectory going from f_start to b_start, passing through f_common ==(state) b_common.
    """
    # todo if this is slow, can build the Trajectory directly without creating nodes
    assert f_common.state == b_common.state
    f_node = f_common
    b_node = b_common
    prev_b_node = b_node
    b_node = b_node.parent
    # todo still not correct!
    while b_node:
        new_f_node = node_type(
            state=b_node.state,
            parent=f_node,
            action=prev_b_node.reverse_action,
            g_cost=f_node.g_cost + 1,
        )
        f_node = new_f_node
        prev_b_node = b_node
        b_node = b_node.parent

    return Trajectory(f_node, num_expanded, device=device)


class MergedTrajectory:
    def __init__(self, trajs: list, shuffle: bool = False):
        if trajs:
            self.states = to.cat(tuple(t.states for t in trajs))
            device = self.states.device
            self.actions = to.cat(tuple(t.actions for t in trajs))
            indices = to.arange(len(trajs), device=device)
            self.indices = to.repeat_interleave(
                indices, to.tensor(tuple(len(t) for t in trajs), device=device)
            )
            self.nums_expanded = to.tensor(
                tuple(t.num_expanded for t in trajs), dtype=to.float32, device=device
            )
            self.num_trajs = len(self.nums_expanded)
            self.num_states = len(self.states)

            if shuffle:
                self.shuffle()
        else:
            return None

    def __len__(self):
        raise NotImplementedError

    def shuffle(self):
        device = self.states.device
        perm = to.randperm(self.num_states, device=device)
        self.states = self.states[perm]
        self.actions = self.actions[perm]
        self.indices = self.indices[perm]
