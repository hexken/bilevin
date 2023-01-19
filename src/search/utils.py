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


class Trajectory:
    def __init__(
        self,
        problem,
        final_node: SearchNode,
        num_expanded: int,
    ):
        """
        Receives a SearchNode representing a solution to the problem.
        Backtracks the path performed by search, collecting state-action pairs along the way.
        """
        self.num_expanded = num_expanded

        action = final_node.parent_action
        node = final_node.parent
        states = []
        actions = []
        cost_to_gos = []
        cost = 1

        while node:
            state_t = problem.state_tensor(node.state)
            states.append(state_t)
            actions.append(action)
            cost_to_gos.append(cost)
            action = node.parent_action
            node = node.parent
            cost += 1

        self.states = to.stack(states[::-1])
        self.actions = to.tensor(actions[::-1])
        self.cost_to_gos = to.tensor(cost_to_gos[::-1])

    def __len__(self):
        return len(self.states)


class MergedTrajectory:
    def __init__(self, trajs: list, shuffle: bool = False):
        if trajs:
            self.states = to.cat(tuple(t.states for t in trajs))
            self.actions = to.cat(tuple(t.actions for t in trajs))
            indices = to.arange(len(trajs))
            self.indices = to.repeat_interleave(
                indices, to.tensor(tuple(len(t) for t in trajs))
            )
            self.nums_expanded = to.tensor(
                tuple(t.num_expanded for t in trajs), dtype=to.float32
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
        perm = to.randperm(self.num_states)
        self.states = self.states[perm]
        self.actions = self.actions[perm]
        self.indices = self.indices[perm]
