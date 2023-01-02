import copy
import math

import numpy as np
import torch as to

from domains.domain import Domain, State
from enums import FourDir


class SlidingTilePuzzleState(State):
    def __init__(
        self, tiles: np.ndarray, blank_pos: tuple[int, int], width: int, num_tiles: int
    ):
        self.tiles = tiles
        self.blank_pos = blank_pos
        self.width = width
        self.num_tiles = num_tiles

    def as_tensor(self, device=to.device("cpu")):

        arr = to.zeros((self.num_tiles, self.width, self.width), device=device)
        arr[:, self.tiles] = 1
        x=2
        return arr

    def __repr__(self):
        return self.one_line()

    def __hash__(self):
        return hash(self.tiles.tobytes())

    def __eq__(self, other):
        return np.array_equal(self.tiles, other.tiles)

    def one_line(self):
        return " ".join(str(t) for t in self.tiles)

    def print(self):
        for i in range(len(self.tiles)):
            print(self.tiles[i], end=" ")
            if (i + 1) % self.width == 0:
                print()


class SlidingTilePuzzle(Domain):
    def __init__(self, tiles: str):
        tiles_l = tiles.replace("\n", "").split(" ")
        self.num_tiles = len(tiles_l)
        self.width = int(math.sqrt(self.num_tiles))
        tiles_arr = np.zeros((self.width, self.width), dtype=np.int32)

        for i, tile in enumerate(tiles_l):
            row = i // self.width
            col = i % self.width
            val = int(tile)
            tiles_arr[row, col] = val

            if val == 0:
                blank_pos = (row, col)

        self.initial_state = SlidingTilePuzzleState(
            tiles_arr, blank_pos, self.width, self.num_tiles  # type:ignore
        )

        self.goal = np.arange(self.num_tiles, dtype=np.int32).reshape(
            self.width, self.width
        )

    def reset(self):
        return self.initial_state

    @property
    def state_size(self):
        return self.width

    @property
    def num_actions(cls):
        return 4

    @property
    def in_channels(self):
        return self.num_tiles

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
        problem = copy.deepcopy(self)
        problem.goal = problem.initial_state.tiles
        problem.initial_state = SlidingTilePuzzleState(
            problem.goal, (0, 0), problem.width, problem.num_tiles
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

        if action == FourDir.UP:
            state.tiles[blank_pos], state.tiles[blank_row - 1, blank_col] = (
                state.tiles[blank_row - 1, blank_col],
                state.tiles[blank_pos],
            )
            state.blank_pos = (blank_row - 1, blank_col)

        elif action == FourDir.DOWN:
            state.tiles[blank_pos], state.tiles[blank_row + 1, blank_col] = (
                state.tiles[blank_row + 1, blank_col],
                state.tiles[blank_pos],
            )
            state.blank_pos = (blank_row + 1, blank_col)

        elif action == FourDir.RIGHT:
            state.tiles[blank_pos], state.tiles[blank_row, blank_col + 1] = (
                state.tiles[blank_row, blank_col + 1],
                state.tiles[blank_pos],
            )
            state.blank_pos = (blank_row, blank_col + 1)

        elif action == FourDir.LEFT:
            state.tiles[blank_pos], state.tiles[blank_row, blank_col - 1] = (
                state.tiles[blank_row, blank_col - 1],
                state.tiles[blank_pos],
            )
            state.blank_pos = (blank_row, blank_col - 1)

    def is_goal(self, state: SlidingTilePuzzleState):
        return np.array_equal(state.tiles, self.goal)

    def try_make_solution(self, state: SlidingTilePuzzleState, other_problem):
        pass
