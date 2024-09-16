from __future__ import annotations

import numpy as np
import torch as to
from torch import from_numpy
from torch.nn.functional import one_hot

from domains.domain import Domain
from domains.state import State
from enums import ActionDir


class SlidingTileState(State):
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
        self._hash = tiles.tobytes().__hash__()

    def __repr__(self) -> str:
        mlw = self.tiles.shape[0] ** 2
        return (
            f" {np.array2string(self.tiles, separator=' ' , max_line_width=mlw)[1:-1]}"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: SlidingTileState) -> bool:
        return (self.tiles == other.tiles).all()


# TSlidingTileState = TypeVar('TSlidingTileState', bound=SlidingTileState)


class SlidingTile(Domain):
    def __init__(
        self,
        start_state: SlidingTileState,
        goal_state: SlidingTileState | None = None,
        forward: bool = True,
    ):
        super().__init__(forward=forward)

        self.start_state: SlidingTileState = start_state
        self.width: int
        self.num_tiles: int

        self.goal_state: SlidingTileState | None = goal_state
        self.goal_state_t: to.Tensor | None = None

    def init(self) -> SlidingTileState | list[SlidingTileState]:
        self.width = self.start_state.tiles.shape[0]
        self.num_tiles = self.width**2

        if self.goal_state is not None:
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
        state: SlidingTileState,
    ) -> to.Tensor:
        return one_hot(from_numpy(state.tiles)).double().permute(2, 0, 1)

    def reverse_action(self, action: ActionDir) -> ActionDir:
        if action == ActionDir.UP:
            return ActionDir.DOWN
        elif action == ActionDir.DOWN:
            return ActionDir.UP
        elif action == ActionDir.LEFT:
            return ActionDir.RIGHT
        elif action == ActionDir.RIGHT:
            return ActionDir.LEFT

    def backward_domain(self) -> SlidingTile:
        assert self.forward
        domain = SlidingTile(self.start_state, self.goal_state, forward=False)
        return domain

    def actions(
        self, parent_action: ActionDir | None, state: SlidingTileState
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

    def actions_unpruned(self, state: SlidingTileState) -> list[ActionDir]:
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

    def result(self, state: SlidingTileState, action: ActionDir) -> SlidingTileState:
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

        new_state = SlidingTileState(tiles, blank_row, blank_col)
        return new_state

    def is_goal(self, state: SlidingTileState) -> bool:
        return state == self.goal_state


def is_solvable(tiles: np.ndarray) -> bool:
    """check if sate is solvable with respect to the canonical goal state"""
    n = tiles.shape[0]
    flat_tiles = tiles.flatten()
    flat_tiles = flat_tiles[flat_tiles != 0]
    inversions = 0
    for i in range(len(flat_tiles)):
        for j in range(i + 1, len(flat_tiles)):
            if flat_tiles[i] > flat_tiles[j]:
                inversions += 1
    if n % 2 == 1:
        return inversions % 2 == 0
    else:
        blank_row_from_top = np.where(tiles == 0)[0][0] + 1
        return (blank_row_from_top % 2 == 0) != (inversions % 2 == 0)


def get_canonical_goal_state(width: int) -> SlidingTileState:
    tiles = np.arange(width**2).reshape(width, width)
    return SlidingTileState(tiles, 0, 0)


def get_permutation(width: int, rng) -> SlidingTileState:
    """get a random stp state that is solvable wrp to the canonical goal state"""
    while True:
        if rng is None:
            tiles = np.random.permutation(width**2).reshape(width, width)
        else:
            tiles = rng.permutation(width**2).reshape(width, width)
        if is_solvable(tiles):
            break
    r, c = np.where(tiles == 0)
    r = r.item()
    c = c.item()
    state = SlidingTileState(tiles, r, c)
    return state
