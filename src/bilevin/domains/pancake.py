from __future__ import annotations

import numpy as np
import torch as to
from torch import from_numpy, transpose
from torch.nn.functional import one_hot

from domains.domain import Domain, State
from enums import ActionDir


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
        # don't include flipping the top pancake, which does nothing
        return self.num_pancakes - 1

    @property
    def in_channels(self) -> int:
        return self.num_pancakes

    def state_tensor(
        self,
        state: PancakePuzzleState,
    ) -> to.Tensor:
        t =  transpose(one_hot(from_numpy(state.pancakes)).float(), 1, 0)
        return t

    def reverse_action(self, action: int) -> int:
        return action

    def backward_domain(self) -> PancakePuzzle:
        assert self.forward
        domain = PancakePuzzle(get_goal_state(self.num_pancakes), forward=False)
        domain.goal_state = self.initial_state
        domain.goal_state_t = self.state_tensor(self.initial_state)
        return domain

    def _actions(self, parent_action: int, state: PancakePuzzleState) -> list[int]:
        return [i for i in range(self.num_actions) if i != parent_action]

    def _actions_unpruned(self, state: PancakePuzzleState):
        return [i for i in range(self.num_actions)]

    def result(self, state: PancakePuzzleState, action: ActionDir) -> PancakePuzzleState:
        pancakes = state.pancakes.copy()
        pancakes[action:] = np.flip(state.pancakes[action:])
        new_state = PancakePuzzleState(pancakes)
        return new_state

    def is_goal(self, state: PancakePuzzleState) -> bool:
        return state == self.goal_state
