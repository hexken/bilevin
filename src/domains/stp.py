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

import numpy as np
import torch as to

from domains.domain import Domain, State, Problem
from enums import FourDir
from search.utils import SearchNode, Trajectory


class SlidingTilePuzzleState(State):
    def __init__(
        self,
        tiles: np.ndarray,
        blank_row: int,
        blank_col: int,
    ):
        self.tiles = tiles
        self.blank_row = blank_row
        self.blank_col = blank_col

    def __repr__(self) -> str:
        mlw = self.tiles.shape[0] ** 2
        return (
            f" {np.array2string(self.tiles, separator=' ' , max_line_width=mlw)[1:-1]}"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> int:
        return self.tiles.tobytes().__hash__()

    def __eq__(self, other) -> bool:
        return np.array_equal(self.tiles, other.tiles)


class SlidingTilePuzzle(Domain):
    def __init__(self, init_tiles: np.ndarray, goal_tiles: np.ndarray, forward=True):
        self.width = init_tiles.shape[0]
        self.num_tiles = self.width**2

        self.forward = forward

        width_indices = np.arange(self.width)
        self._row_indices = np.repeat(width_indices, self.width)
        self._col_indices = np.tile(width_indices, self.width)

        blank_pos = np.where(init_tiles == 0)
        self.blank_row = blank_pos[0].item()
        self.blank_col = blank_pos[1].item()

        self.initial_state = SlidingTilePuzzleState(
            init_tiles, self.blank_row, self.blank_col
        )

        self.initial_state_t = self.state_tensor(self.initial_state)

        if not self.forward:
            self.state_tensor = self.backward_state_tensor

        self.goal_tiles = goal_tiles

        self.visited = {}

    def reset(self) -> SlidingTilePuzzleState:
        self.visited = {}
        return self.initial_state

    def update(self, node: SearchNode):
        self.visited[node.state.__hash__()] = node

    @property
    def state_width(self) -> int:
        return self.width

    @property
    def num_actions(cls) -> int:
        return 4

    @property
    def in_channels(self) -> int:
        return self.num_tiles

    def state_tensor(
        self,
        state: SlidingTilePuzzleState,
    ) -> to.Tensor:
        arr = np.zeros((self.num_tiles, self.width, self.width), dtype=np.float32)
        indices = state.tiles.reshape(-1)
        arr[
            indices,
            self._row_indices,
            self._col_indices,
        ] = 1
        return to.from_numpy(arr)

    def backward_state_tensor(
        self,
        state: SlidingTilePuzzleState,
    ) -> to.Tensor:
        arr = np.zeros((self.num_tiles, self.width, self.width), dtype=np.float32)
        arr[
            state.tiles.reshape(-1),
            self._row_indices,
            self._col_indices,
        ] = 1
        return to.stack((to.from_numpy(arr), self.initial_state_t))

    def reverse_action(self, action: FourDir) -> FourDir:
        if action == FourDir.UP:
            return FourDir.DOWN
        elif action == FourDir.DOWN:
            return FourDir.UP
        elif action == FourDir.LEFT:
            return FourDir.RIGHT
        elif action == FourDir.RIGHT:
            return FourDir.LEFT

    def backward_domain(self) -> SlidingTilePuzzle:
        assert self.forward
        init_tiles = self.goal_tiles
        goal_tiles = self.initial_state.tiles
        domain = SlidingTilePuzzle(init_tiles, goal_tiles, False)

        return domain

    def actions(
        self, parent_action: FourDir, state: SlidingTilePuzzleState
    ) -> list[FourDir]:
        return self.actions_unpruned(state)

    def actions_unpruned(self, state: SlidingTilePuzzleState):
        actions = []

        if state.blank_col != self.width - 1:
            actions.append(FourDir.RIGHT)

        if state.blank_row != 0:
            actions.append(FourDir.UP)

        if state.blank_col != 0:
            actions.append(FourDir.LEFT)

        if state.blank_row != self.width - 1:
            actions.append(FourDir.DOWN)

        return actions

    def result(
        self, state: SlidingTilePuzzleState, action: FourDir
    ) -> SlidingTilePuzzleState:
        tiles = np.array(state.tiles)
        blank_row = state.blank_row
        blank_col = state.blank_col

        if action == FourDir.UP:
            tiles[blank_row, blank_col], tiles[blank_row - 1, blank_col] = (
                tiles[blank_row - 1, blank_col],
                tiles[blank_row, blank_col],
            )
            blank_row -= 1

        elif action == FourDir.DOWN:
            tiles[blank_row, blank_col], tiles[blank_row + 1, blank_col] = (
                tiles[blank_row + 1, blank_col],
                tiles[blank_row, blank_col],
            )
            blank_row += 1

        elif action == FourDir.RIGHT:
            tiles[blank_row, blank_col], tiles[blank_row, blank_col + 1] = (
                tiles[blank_row, blank_col + 1],
                tiles[blank_row, blank_col],
            )
            blank_col += 1

        elif action == FourDir.LEFT:
            tiles[blank_row, blank_col], tiles[blank_row, blank_col - 1] = (
                tiles[blank_row, blank_col - 1],
                tiles[blank_row, blank_col],
            )
            blank_col -= 1

        new_state = SlidingTilePuzzleState(tiles, blank_row, blank_col)
        return new_state

    def is_goal(self, state: SlidingTilePuzzleState) -> bool:
        return np.array_equal(state.tiles, self.goal_tiles)

    def get_merged_trajectory(
        self,
        dir1_common: SearchNode,
        dir2_common: SearchNode,
        node_type: Type[SearchNode],
        num_expanded: int,
    ):
        """
        Returns a new trajectory going from dir1_start to dir2_start, passing through
        merge(dir1_common, dir2_common).
        """
        dir1_node = dir1_common
        parent_dir2_node = dir2_common.parent
        parent_dir2_action = dir2_common.parent_action
        while parent_dir2_node:
            new_dir1_node = node_type(
                state=parent_dir2_node.state,
                parent=dir1_node,
                parent_action=self.reverse_action(parent_dir2_action),
                g_cost=dir1_node.g_cost + 1,
            )
            dir1_node = new_dir1_node
            parent_dir2_action = parent_dir2_node.parent_action
            parent_dir2_node = parent_dir2_node.parent

        return Trajectory(self, dir1_node, num_expanded)

    def try_make_solution(
        self,
        node: SearchNode,
        other_domain: SlidingTilePuzzle,
        num_expanded: int,
    ) -> Optional[tuple[Trajectory, Trajectory]]:
        """
        Returns a trajectory if state is a solution to this problem, None otherwise.
        """
        hsh = node.state.__hash__()
        if hsh in other_domain.visited:  # solution found
            other_node = other_domain.visited[hsh]
            if self.forward:
                f_common_node = node
                b_common_node = other_node
                f_domain = self
                b_domain = other_domain
            else:
                f_common_node = other_node
                b_common_node = node
                f_domain = other_domain
                b_domain = self

            f_traj = f_domain.get_merged_trajectory(
                f_common_node, b_common_node, type(node), num_expanded
            )
            b_traj = b_domain.get_merged_trajectory(
                b_common_node, f_common_node, type(node), num_expanded
            )

            return (f_traj, b_traj)
        else:
            return None


def load_problemset(problemset: dict):
    width = problemset["width"]
    problems = []
    goal_tiles = np.arange(width**2).reshape(width, width)
    for p_dict in problemset["problems"]:
        init_tiles = np.array(p_dict["tiles"])
        problem = Problem(
            id=p_dict["id"],
            domain=SlidingTilePuzzle(init_tiles=init_tiles, goal_tiles=goal_tiles),
        )
        problems.append(problem)

    num_actions = problems[0].domain.num_actions
    in_channels = problems[0].domain.in_channels
    state_t_width = problems[0].domain.state_width
    double_backward = True

    return problems, num_actions, in_channels, state_t_width, double_backward
