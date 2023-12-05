from __future__ import annotations

import numpy as np
import torch as to

from domains.domain import Domain, State
from enums import FourDir


def get_goal_state(n_pancakes: int) -> PancakePuzzleState:
    pancakes = np.arange(n_pancakes)
    return PancakePuzzleState(pancakes)


class PancakePuzzleState(State):
    def __init__(
        self,
        pancakes: np.ndarray,
    ):
        self.pancakes: np.ndarray = pancakes

    def __repr__(self) -> str:
        return self.pancakes.__repr__()

    def __str__(self) -> str:
        return self.pancakes.__str__()

    def __hash__(self) -> int:
        return self.pancakes.tobytes().__hash__()

    def __eq__(self, other: PancakePuzzleState) -> bool:
        return (self.pancakes == other.pancakes).all()


class PancakePuzzle(Domain):
    def __init__(self, initial_state: PancakePuzzleState, forward: bool = True):
        super().__init__(forward=forward)

        self.initial_state: PancakePuzzleState = initial_state
        self.num_pancakes: int = len(initial_state.pancakes)

        self.goal_state: PancakePuzzleState
        self.goal_state_t: to.Tensor

    def reset(self) -> State:
        if self.forward:
            self.goal_state = get_goal_state(self.num_pancakes)
            self.goal_state_t = self.state_tensor(self.goal_state)
        return self._reset()

    @property
    def state_t_width(self) -> int:
        return self.num_pancakes

    @property
    def state_t_depth(self) -> int:
        return 1

    @property
    def num_actions(self) -> int:
        return self.num_pancakes

    @property
    def in_channels(self) -> int:
        return self.num_pancakes

    def state_tensor(
        self,
        state: PancakePuzzleState,
    ) -> to.Tensor:
        arr = np.zeros((self.num_pancakes, self.num_pancakes), dtype=np.float32)
        indices = state.pancakes.reshape(-1)
        arr[indices,] = 1
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

    def backward_domain(self) -> PancakePuzzle:
        assert self.forward
        domain = PancakePuzzle(get_goal_state(self.width), forward=False)
        domain.goal_state = self.initial_state
        domain.goal_state_t = self.state_tensor(self.initial_state)
        return domain

    def _actions(
        self, parent_action: FourDir, state: PancakePuzzleState
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

    def _actions_unpruned(self, state: PancakePuzzleState):
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

    def result(self, state: PancakePuzzleState, action: FourDir) -> PancakePuzzleState:
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

        new_state = PancakePuzzleState(tiles, blank_row, blank_col)
        return new_state

    def is_goal(self, state: PancakePuzzleState) -> bool:
        return state == self.goal_state
