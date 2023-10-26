from __future__ import annotations

import numpy as np
import torch as to

from domains.domain import Domain, State
from enums import FourDir


def get_goal_state(width: int) -> SlidingTilePuzzleState:
    tiles = np.arange(width**2).reshape(width, width)
    return SlidingTilePuzzleState(tiles, 0, 0)


class SlidingTilePuzzleState(State):
    def __init__(
        self,
        tiles: np.ndarray,
        blank_row: int,
        blank_col: int,
    ):
        self.tiles: np.ndarray = tiles
        self.blank_row: int = blank_row
        self.blank_col: int = blank_col
        self.width = len(tiles)

    def __repr__(self) -> str:
        mlw = self.tiles.shape[0] ** 2
        return (
            f" {np.array2string(self.tiles, separator=' ' , max_line_width=mlw)[1:-1]}"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> int:
        return self.tiles.tobytes().__hash__()

    def __eq__(self, other: SlidingTilePuzzleState) -> bool:
        return (self.tiles == other.tiles).all()


class SlidingTilePuzzle(Domain):
    def __init__(self, initial_state: SlidingTilePuzzleState, forward: bool = True):
        super().__init__(forward=forward)

        self.initial_state: SlidingTilePuzzleState = initial_state
        self.width: int
        self.num_tiles: int

        self._row_indices: np.ndarray
        self._col_indices = np.ndarray

        self.goal_state: SlidingTilePuzzleState
        self.goal_state_t: to.Tensor

    def reset(self) -> State:
        self.width = self.initial_state.tiles.shape[0]
        self.num_tiles = self.width**2

        width_indices = np.arange(self.width)
        self._row_indices = np.repeat(width_indices, self.width)
        self._col_indices = np.tile(width_indices, self.width)

        self.goal_state = get_goal_state(self.width)
        self.goal_state_t = self.state_tensor(self.goal_state)
        return self._reset()

    @property
    def state_width(self) -> int:
        return self.width

    @property
    def requires_backward_goal(self):
        return True

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
        init_state = get_goal_state(self.width)
        domain = SlidingTilePuzzle(init_state, forward=False)
        domain.goal_state = self.initial_state
        domain.goal_state_t = self.state_tensor(init_state)

        return domain

    def _actions(
        self, parent_action: FourDir, state: SlidingTilePuzzleState
    ) -> list[FourDir]:
        actions = []

        if parent_action != FourDir.LEFT and state.blank_col != self.width - 1:
            actions.append(FourDir.RIGHT)

        if parent_action != FourDir.DOWN and state.blank_row != 0:
            actions.append(FourDir.UP)

        if parent_action != FourDir.RIGHT and state.blank_col != 0:
            actions.append(FourDir.LEFT)

        if parent_action != FourDir.UP and state.blank_row != self.width - 1:
            actions.append(FourDir.DOWN)

        return actions

    def _actions_unpruned(self, state: SlidingTilePuzzleState):
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
        return state == self.goal_state
