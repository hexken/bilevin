from __future__ import annotations

import numpy as np
import torch as to
from torch import from_numpy, transpose
from torch.nn.functional import one_hot

from domains.domain import Domain
from domains.state import State
from enums import ActionDir


class PancakeState(State):
    def __init__(
        self,
        pancakes: np.ndarray,
    ):
        self.pancakes: np.ndarray = pancakes
        self._hash = pancakes.tobytes().__hash__()

    def __repr__(self) -> str:
        return self.pancakes.__repr__()

    def __str__(self) -> str:
        return self.pancakes.__str__()

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: PancakeState) -> bool:
        return (self.pancakes == other.pancakes).all()


class Pancake(Domain):
    def __init__(
        self,
        start_state: PancakeState,
        goal_state: PancakeState | None = None,
        forward: bool = True,
    ):
        super().__init__(forward=forward)

        self.start_state: PancakeState = start_state
        self.num_pancakes: int = len(start_state.pancakes)

        self.goal_state: PancakeState | None = goal_state
        self.goal_state_t: to.Tensor | None = None

    def init(self) -> PancakeState | list[PancakeState]:
        if self.goal_state is not None:
            self.goal_state_t = self.state_tensor(self.goal_state)
        return self._init()

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
        state: PancakeState,
    ) -> to.Tensor:
        t = transpose(one_hot(from_numpy(state.pancakes)).double(), 1, 0).flatten()
        return t

    def reverse_action(self, action: int) -> int:
        return action

    def backward_domain(self) -> Pancake:
        assert self.forward
        domain = Pancake(self.goal_state, self.start_state, forward=False)
        return domain

    def actions(self, parent_action: int | None, state: PancakeState) -> list[int]:
        return [i for i in range(self.num_actions) if i != parent_action]

    def actions_unpruned(self, state: PancakeState):
        return [i for i in range(self.num_actions)]

    def result(self, state: PancakeState, action: ActionDir) -> PancakeState:
        pancakes = state.pancakes.copy()
        pancakes[action:] = np.flip(state.pancakes[action:])
        new_state = PancakeState(pancakes)
        return new_state

    def is_goal(self, state: PancakeState) -> bool:
        return state == self.goal_state


def get_canonical_goal_state(n_pancakes: int) -> PancakeState:
    pancakes = np.arange(n_pancakes)
    return PancakeState(pancakes)


def get_permutation(n_pancakes: int, rng=None) -> PancakeState:
    if rng is None:
        pancakes = np.random.permutation(n_pancakes)
    else:
        pancakes = rng.permutation(n_pancakes)
    return PancakeState(pancakes)
