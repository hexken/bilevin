# Copyright (C) 2021-2022, Ken Tjhia
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations
from typing import Optional, Type

import torch as to

from domains.domain import State, Domain


class SearchNode:
    def __init__(
        self,
        state: State,
        g_cost: float,
        parent: Optional[SearchNode] = None,
        parent_action: Optional[int] = None,
    ):
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
        domain: Domain,
        final_node: SearchNode,
        num_expanded: int,
        goal_state_t: Optional[to.Tensor] = None,
    ):
        """
        Receives a SearchNode representing a solution to the problem.
        Backtracks the path performed by search, collecting state-action pairs along the way.
        """
        self.num_expanded = num_expanded
        self.goal_state_t = (
            goal_state_t.unsqueeze(0) if goal_state_t is not None else None
        )

        action = final_node.parent_action
        node = final_node.parent
        states = []
        actions = []

        while node:
            state_t = domain.state_tensor(node.state)
            states.append(state_t)
            actions.append(action)
            action = node.parent_action
            node = node.parent

        self.states = to.stack(states[::-1])
        self.actions = to.tensor(actions[::-1])
        self.cost_to_gos = to.arange(len(self.states), 0, -1)

    def __len__(self):
        return len(self.states)


class MergedTrajectory:
    def __init__(self, trajs: list):
        if trajs:
            if trajs[0].goal_state_t is not None:
                self.goal_states = to.cat(tuple(t.goal_state_t for t in trajs))
            else:
                self.goal_states = None

            self.states = to.cat(tuple(t.states for t in trajs))
            self.actions = to.cat(tuple(t.actions for t in trajs))
            indices = to.arange(len(trajs))
            self.lengths = to.tensor(tuple(len(t) for t in trajs))
            self.indices = to.repeat_interleave(indices, self.lengths)
            self.nums_expanded = to.tensor(
                tuple(t.num_expanded for t in trajs), dtype=to.float32
            )
            self.num_trajs = len(self.nums_expanded)
            self.num_states = len(self.states)

            # if shuffle:
            #     self.shuffle()
        else:
            return None

    def __len__(self):
        raise NotImplementedError

    # def shuffle(self):
    #     perm = to.randperm(self.num_states)
    #     self.states = self.states[perm]
    #     self.actions = self.actions[perm]
    #     self.indices = self.indices[perm]
