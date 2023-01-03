from typing import Type

import torch as to


class SearchNode:
    def __init__(self, state, parent, parent_action, g_cost):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.g_cost = g_cost

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


class BidirectionalSearchNode(SearchNode):
    def __init__(self, state, parent, action, g_cost):
        super().__init__(state, parent, action, g_cost)


class Trajectory:
    def __init__(
        self,
        final_node: SearchNode,
        num_expanded: int,
        device: to.device = to.device("cpu"),
    ):
        """
        Receives a SearchNode representing a solution to the problem.
        Backtracks the path performed by search, collecting state-action pairs along the way.
        """
        self.device = device
        self.num_expanded = num_expanded
        # if hasattr(final_node, "log_prob"):
        #     self.solution_prob = exp(final_node.log_prob)  # type:ignore

        action = final_node.parent_action
        node = final_node.parent
        # self.goal = final_node.state.as_tensor(device)
        states = []
        actions = []
        cost_to_gos = []
        cost = 1

        while node:
            states.append(node.state.as_tensor(device))
            actions.append(action)
            cost_to_gos.append(cost)
            action = node.parent_action
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
