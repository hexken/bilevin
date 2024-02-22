from __future__ import annotations

import numpy as np
import torch as to
from torch import from_numpy
from torch.nn.functional import one_hot

from domains.domain import Domain, State
from enums import ActionDir


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

        self.goal_state: SlidingTilePuzzleState
        self.goal_state_t: to.Tensor

    def init(self) -> State:
        self.width = self.initial_state.tiles.shape[0]
        self.num_tiles = self.width**2

        if self.forward:
            self.goal_state = get_goal_state(self.width)
            self.goal_state_t = self.state_tensor(self.goal_state)
        return self._init()

    @property
    def state_t_width(self) -> int:
        return self.width

    @property
    def state_t_depth(self) -> int:
        return 1

    @property
    def num_actions(self) -> int:
        return 4

    @property
    def in_channels(self) -> int:
        return self.num_tiles

    def state_tensor(
        self,
        state: SlidingTilePuzzleState,
    ) -> to.Tensor:
        return one_hot(from_numpy(state.tiles)).float().permute(2, 0, 1)

    def reverse_action(self, action: ActionDir) -> ActionDir:
        if action == ActionDir.UP:
            return ActionDir.DOWN
        elif action == ActionDir.DOWN:
            return ActionDir.UP
        elif action == ActionDir.LEFT:
            return ActionDir.RIGHT
        elif action == ActionDir.RIGHT:
            return ActionDir.LEFT

    def backward_domain(self) -> SlidingTilePuzzle:
        assert self.forward
        domain = SlidingTilePuzzle(get_goal_state(self.width), forward=False)
        domain.goal_state = self.initial_state
        domain.goal_state_t = self.state_tensor(self.initial_state)
        return domain

    def _actions(
        self, parent_action: ActionDir, state: SlidingTilePuzzleState
    ) -> list[ActionDir]:
        actions = []

        if parent_action != ActionDir.LEFT and state.blank_col != self.width - 1:
            actions.append(ActionDir.RIGHT)

        if parent_action != ActionDir.DOWN and state.blank_row != 0:
            actions.append(ActionDir.UP)

        if parent_action != ActionDir.RIGHT and state.blank_col != 0:
            actions.append(ActionDir.LEFT)

        if parent_action != ActionDir.UP and state.blank_row != self.width - 1:
            actions.append(ActionDir.DOWN)

        return actions

    def _actions_unpruned(self, state: SlidingTilePuzzleState):
        actions = []

        if state.blank_col != self.width - 1:
            actions.append(ActionDir.RIGHT)

        if state.blank_row != 0:
            actions.append(ActionDir.UP)

        if state.blank_col != 0:
            actions.append(ActionDir.LEFT)

        if state.blank_row != self.width - 1:
            actions.append(ActionDir.DOWN)

        return actions

    def result(
        self, state: SlidingTilePuzzleState, action: ActionDir
    ) -> SlidingTilePuzzleState:
        tiles = np.array(state.tiles)
        blank_row = state.blank_row
        blank_col = state.blank_col

        if action == ActionDir.UP:
            tiles[blank_row, blank_col], tiles[blank_row - 1, blank_col] = (
                tiles[blank_row - 1, blank_col],
                tiles[blank_row, blank_col],
            )
            blank_row -= 1

        elif action == ActionDir.DOWN:
            tiles[blank_row, blank_col], tiles[blank_row + 1, blank_col] = (
                tiles[blank_row + 1, blank_col],
                tiles[blank_row, blank_col],
            )
            blank_row += 1

        elif action == ActionDir.RIGHT:
            tiles[blank_row, blank_col], tiles[blank_row, blank_col + 1] = (
                tiles[blank_row, blank_col + 1],
                tiles[blank_row, blank_col],
            )
            blank_col += 1

        elif action == ActionDir.LEFT:
            tiles[blank_row, blank_col], tiles[blank_row, blank_col - 1] = (
                tiles[blank_row, blank_col - 1],
                tiles[blank_row, blank_col],
            )
            blank_col -= 1

        new_state = SlidingTilePuzzleState(tiles, blank_row, blank_col)
        return new_state

    def is_goal(self, state: SlidingTilePuzzleState) -> bool:
        return state == self.goal_state
