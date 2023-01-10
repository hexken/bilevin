from __future__ import annotations
from copy import deepcopy
import math
from typing import Optional, Type

import numpy as np
import torch as to

from domains.domain import Domain, State
from enums import FourDir
from search.utils import SearchNode, Trajectory


class SlidingTilePuzzleState(State):
    def __init__(
        self,
        tiles: np.ndarray,
        blank_pos: tuple[int, int],
        width: int,
        num_tiles: int,
    ):
        self.tiles = tiles
        self.blank_pos = blank_pos
        self.width = width
        self.num_tiles = num_tiles

    def __repr__(self):
        return self.oneline()

    def __hash__(self):
        return self.tiles.tobytes().__hash__()

    def __eq__(self, other):
        return np.array_equal(self.tiles, other.tiles)

    def oneline(self) -> str:
        return " ".join(str(t) for t in self.tiles.flatten())

    def print(self):
        for i in range(len(self.tiles)):
            print(self.tiles[i], end=" ")
            if (i + 1) % self.width == 0:
                print()


class SlidingTilePuzzle(Domain):
    def __init__(self, tiles: str | np.ndarray):
        if type(tiles) == str:
            tiles_l = tiles.replace("\n", "").split(" ")
            self.num_tiles = len(tiles_l)
            self.width = int(math.sqrt(self.num_tiles))

            _tiles = np.arange(self.width)

            tiles_arr = np.zeros((self.width, self.width), dtype=np.int32)

            for i, tile in enumerate(tiles_l):
                row = i // self.width
                col = i % self.width
                val = int(tile)
                tiles_arr[row, col] = val

                if val == 0:
                    blank_pos = (row, col)
        elif type(tiles) == np.ndarray:
            self.width = len(tiles)
            _tiles = np.arange(self.width)
            self.num_tiles = self.width**2
            tiles_arr = tiles
            blank_pos = np.where(tiles_arr == 0)
            blank_pos = blank_pos[0].item(), blank_pos[1].item()
        else:
            raise ValueError("Invalid tiles spec")

        self._row_indices = np.repeat(_tiles, self.width)
        self._col_indices = np.tile(_tiles, self.width)
        self.initial_state = SlidingTilePuzzleState(
            tiles_arr, blank_pos, self.width, self.num_tiles  # type:ignore
        )
        self.forward = True
        self.initial_state_t = self.state_tensor(self.initial_state)

        self.goal = np.arange(self.num_tiles, dtype=np.int32).reshape(
            self.width, self.width
        )

        self.visited = {}

    def reset(self):
        self.visited = {}
        return self.initial_state

    def update(self, node: SearchNode):
        self.visited[node.state.__hash__()] = node

    @property
    def state_width(self):
        return self.width

    @property
    def num_actions(cls):
        return 4

    @property
    def in_channels(self):
        return self.num_tiles

    def state_tensor(
        self,
        state: SlidingTilePuzzleState,
    ) -> to.Tensor | tuple[to.Tensor, to.Tensor]:
        arr = to.zeros((self.num_tiles, self.width, self.width))
        arr[
            state.tiles.reshape(-1),
            self._row_indices,
            self._col_indices,
        ] = 1

        if self.forward:
            return arr
        else:
            return to.stack(
                (arr, self.initial_state_t)  # type:ignore
            )

    def reverse_action(self, action: FourDir):
        if action == FourDir.UP:
            return FourDir.DOWN
        elif action == FourDir.DOWN:
            return FourDir.UP
        elif action == FourDir.LEFT:
            return FourDir.RIGHT
        elif action == FourDir.RIGHT:
            return FourDir.LEFT

    def backward_problem(self):
        assert self.forward
        problem = deepcopy(self)
        problem.forward = False
        forward_goal = problem.goal
        problem.goal = problem.initial_state.tiles

        problem.initial_state = SlidingTilePuzzleState(
            forward_goal,
            (0, 0),
            problem.width,
            problem.num_tiles,
        )

        return problem

    def actions(self, parent_action: FourDir, state: SlidingTilePuzzleState):
        return self.actions_unpruned(state)

    def actions_unpruned(self, state: SlidingTilePuzzleState):
        actions = []
        blank_row = state.blank_pos[0]
        blank_col = state.blank_pos[1]

        if blank_col != state.width - 1:
            actions.append(FourDir.RIGHT)

        if blank_row != 0:
            actions.append(FourDir.UP)

        if blank_col != 0:
            actions.append(FourDir.LEFT)

        if blank_row != state.width - 1:
            actions.append(FourDir.DOWN)

        return actions

    def result(self, state: SlidingTilePuzzleState, action: FourDir):
        blank_pos = state.blank_pos
        blank_row = state.blank_pos[0]
        blank_col = state.blank_pos[1]

        new_state = deepcopy(state)

        if action == FourDir.UP:
            new_state.tiles[blank_pos], new_state.tiles[blank_row - 1, blank_col] = (
                new_state.tiles[blank_row - 1, blank_col],
                new_state.tiles[blank_pos],
            )
            new_state.blank_pos = (blank_row - 1, blank_col)

        elif action == FourDir.DOWN:
            new_state.tiles[blank_pos], new_state.tiles[blank_row + 1, blank_col] = (
                new_state.tiles[blank_row + 1, blank_col],
                new_state.tiles[blank_pos],
            )
            new_state.blank_pos = (blank_row + 1, blank_col)

        elif action == FourDir.RIGHT:
            new_state.tiles[blank_pos], new_state.tiles[blank_row, blank_col + 1] = (
                new_state.tiles[blank_row, blank_col + 1],
                new_state.tiles[blank_pos],
            )
            new_state.blank_pos = (blank_row, blank_col + 1)

        elif action == FourDir.LEFT:
            new_state.tiles[blank_pos], new_state.tiles[blank_row, blank_col - 1] = (
                new_state.tiles[blank_row, blank_col - 1],
                new_state.tiles[blank_pos],
            )
            new_state.blank_pos = (blank_row, blank_col - 1)

        return new_state

    def is_goal(self, state: SlidingTilePuzzleState):
        return np.array_equal(state.tiles, self.goal)

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
            new_state = deepcopy(parent_dir2_node.state)

            new_dir1_node = node_type(
                state=new_state,
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
        other_problem: SlidingTilePuzzle,
        num_expanded: int,
    ) -> Optional[tuple[Trajectory, Trajectory]]:
        """
        Returns a trajectory if state is a solution to this problem, None otherwise.
        """
        hsh = node.state.__hash__()
        if hsh in other_problem.visited:
            other_node = other_problem.visited[hsh]
            if self.forward:
                f_common_node = node
                b_common_node = other_node
            else:
                f_common_node = other_node
                b_common_node = node

            f_traj = self.get_merged_trajectory(
                f_common_node, b_common_node, type(node), num_expanded
            )
            b_traj = self.get_merged_trajectory(
                b_common_node, f_common_node, type(node), num_expanded
            )

            return (f_traj, b_traj)
        else:
            return None
