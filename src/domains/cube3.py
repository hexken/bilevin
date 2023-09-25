"""
Based on the implementation found at this nice blog post:
    https://blog.dorianbolivar.com/2021/01/solving-rubik-cycle-programming.html
    https://github.com/dbolivar/programming_challenges_exercises/blob/main/URI_Online_Judge/1232-Rubik_Cycle/1232_numpy.py
"""
from __future__ import annotations

import numpy as np
import torch as to
from torch import from_numpy
from torch.nn.functional import one_hot

from domains.domain import Domain, State
from search.utils import Problem


def get_goal_state() -> Cube3State:
    front = np.full((3, 3), 0, dtype=int)
    up = np.full((3, 3), 1, dtype=int)
    down = np.full((3, 3), 2, dtype=int)
    left = np.full((3, 3), 3, dtype=int)
    right = np.full((3, 3), 4, dtype=int)
    back = np.full((3, 3), 5, dtype=int)
    state = Cube3State(front, up, down, left, right, back)
    return state


class Cube3State(State):
    def __init__(
        self,
        front: np.ndarray,
        up: np.ndarray,
        down: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        back: np.ndarray,
    ):
        """
        0 FRONT GREEN
        1 UP WHITE
        2 DOWN YELLOW
        3 LEFT ORANGE
        4 RIGHT RED
        5 BACK BLUE
        """
        self.front: np.ndarray = front
        self.up: np.ndarray = up
        self.down: np.ndarray = down
        self.left: np.ndarray = left
        self.right: np.ndarray = right
        self.back: np.ndarray = back

    def __str__(self) -> str:
        return "NotImplemented"

    def __hash__(self) -> int:
        return (
            self.front.tobytes().__hash__(),
            self.up.tobytes().__hash__(),
            self.down.tobytes().__hash__(),
            self.left.tobytes().__hash__(),
            self.right.tobytes().__hash__(),
            self.back.tobytes().__hash__(),
        ).__hash__()

    def __eq__2(self, other: Cube3State) -> bool:
        return equal_check(
            self.front,
            self.up,
            self.down,
            self.left,
            self.right,
            self.back,
            other.front,
            other.up,
            other.down,
            other.left,
            other.right,
            other.back,
        )

    def __eq__(self, other: Cube3State) -> bool:
        return (
            (self.front == other.front).all()
            and (self.up == other.up).all()
            and (self.down == other.down).all()
            and (self.left == other.left).all()
            and (self.right == other.right).all()
            and (self.back == other.back).all()
        )
        # for i in range(3):
        #     for j in range(3):
        #         if (
        #             self.front[i, j] != other.front[i, j]
        #             or self.up[i, j] != other.up[i, j]
        #             or self.down[i, j] != other.down[i, j]
        #             or self.left[i, j] != other.left[i, j]
        #             or self.right[i, j] != other.right[i, j]
        #             or self.back[i, j] != other.back[i, j]
        #         ):
        #             return False
        # return True


class Cube3(Domain):
    def __init__(
        self,
        initial_state: Cube3State,
        forward: bool = True,
    ):
        super().__init__(forward=forward)
        self.cube_width: int = 3
        self._actions_list: list[int] = [i for i in range(self.num_actions)]
        self.initial_state: Cube3State = initial_state
        # self._forward_result = self.result

        self.goal_state: Cube3State
        self.goal_state_t: to.Tensor

    @property
    def state_width(self) -> int:
        return 3

    @property
    def requires_backward_goal(self) -> bool:
        return True

    @property
    def num_actions(cls) -> int:
        return 12

    @property
    def in_channels(self) -> int:
        return 6

    def _actions(self, parent_action: int, state: Cube3State) -> list[int]:
        """
        0 FRONT GREEN
        1 UP WHITE
        2 DOWN YELLOW
        3 LEFT ORANGE
        4 RIGHT RED
        5 BACK BLUE
        """
        return [
            i
            for i in range(self.num_actions)
            if i != self.reverse_action(parent_action)
        ]

    def _actions_unpruned(self, state: Cube3State) -> list[int]:
        return self._actions_list

    def reverse_action(self, action: int) -> int:
        if action & 1 == 1:  # odd
            return action - 1
        else:
            return action + 1

    def backward_domain(self) -> Cube3:
        assert self.forward
        domain = Cube3(get_goal_state(), forward=False)
        # domain.result = domain._backwards_result
        domain.goal_state = self.initial_state
        domain.goal_state_t = self.state_tensor(self.initial_state)
        return domain

    def result(
        self,
        state: Cube3State,
        action: int,
    ) -> Cube3State:
        f = state.front.copy()
        u = state.up.copy()
        d = state.down.copy()
        l = state.left.copy()
        r = state.right.copy()
        b = state.back.copy()

        if action == 0:
            f = np.rot90(f, -1)
            u[2, :], l[:, 2], d[0, :], r[:, 0] = (
                np.flip(l[:, 2]).copy(),
                d[0, :].copy(),
                np.flip(r[:, 0]).copy(),
                u[2, :].copy(),
            )
        elif action == 1:
            f = np.rot90(f, 1)
            u[2, :], r[:, 0], d[0, :], l[:, 2] = (
                r[:, 0].copy(),
                np.flip(d[0, :]).copy(),
                l[:, 2].copy(),
                np.flip(u[2, :]).copy(),
            )
        elif action == 2:
            # Face rotate.
            u = np.rot90(u, -1)
            b[2, :], l[0, :], f[0, :], r[0, :] = (
                np.flip(l[0, :]).copy(),
                f[0, :].copy(),
                r[0, :].copy(),
                np.flip(b[2, :]).copy(),
            )
        elif action == 3:
            u = np.rot90(u, 1)
            b[2, :], r[0, :], f[0, :], l[0, :] = (
                np.flip(r[0, :]).copy(),
                f[0, :].copy(),
                l[0, :].copy(),
                np.flip(b[2, :]).copy(),
            )
        elif action == 4:
            d = np.rot90(d, -1)
            f[2, :], l[2, :], b[0, :], r[2, :] = (
                l[2, :].copy(),
                np.flip(b[0, :]).copy(),
                np.flip(r[2, :]).copy(),
                f[2, :].copy(),
            )
        elif action == 5:
            d = np.rot90(d, 1)
            f[2, :], r[2, :], b[0, :], l[2, :] = (
                r[2, :].copy(),
                np.flip(b[0, :]).copy(),
                np.flip(l[2, :]).copy(),
                f[2, :].copy(),
            )
        elif action == 6:
            l = np.rot90(l, -1)
            u[:, 0], b[:, 0], d[:, 0], f[:, 0] = (
                b[:, 0].copy(),
                d[:, 0].copy(),
                f[:, 0].copy(),
                u[:, 0].copy(),
            )
        elif action == 7:
            l = np.rot90(l, 1)
            u[:, 0], f[:, 0], d[:, 0], b[:, 0] = (
                f[:, 0].copy(),
                d[:, 0].copy(),
                b[:, 0].copy(),
                u[:, 0].copy(),
            )
        elif action == 8:
            r = np.rot90(r, -1)
            u[:, 2], f[:, 2], d[:, 2], b[:, 2] = (
                f[:, 2].copy(),
                d[:, 2].copy(),
                b[:, 2].copy(),
                u[:, 2].copy(),
            )
        elif action == 9:
            r = np.rot90(r, 1)
            u[:, 2], b[:, 2], d[:, 2], f[:, 2] = (
                b[:, 2].copy(),
                d[:, 2].copy(),
                f[:, 2].copy(),
                u[:, 2].copy(),
            )
        elif action == 10:
            b = np.rot90(b, -1)
            u[0, :], r[:, 2], d[2, :], l[:, 0] = (
                r[:, 2].copy(),
                np.flip(d[2, :]).copy(),
                l[:, 0].copy(),
                np.flip(u[0, :]).copy(),
            )
        else:
            b = np.rot90(b, 1)
            u[0, :], l[:, 0], d[2, :], r[:, 2] = (
                np.flip(l[:, 0]).copy(),
                d[2, :].copy(),
                np.flip(r[:, 2]).copy(),
                u[0, :].copy(),
            )

        return Cube3State(f, u, d, l, r, b)

    def is_goal(self, state: Cube3State) -> bool:
        return (
            (state.front == 0).all()
            and (state.up == 1).all()
            and (state.down == 2).all()
            and (state.left == 3).all()
            and (state.right == 4).all()
            and (state.back == 5).all()
        )

    def state_tensor(self, state: Cube3State) -> to.Tensor:
        return (
            to.stack(
                (
                    one_hot(from_numpy(state.front.copy()), 6),
                    one_hot(from_numpy(state.up.copy()), 6),
                    one_hot(from_numpy(state.down.copy()), 6),
                    one_hot(from_numpy(state.left.copy()), 6),
                    one_hot(from_numpy(state.right.copy()), 6),
                    one_hot(from_numpy(state.back.copy()), 6),
                )
            )
            .float()
            .permute(3, 0, 1, 2)
        )


def parse_problemset(problemset: dict):
    def parse_specs(problem_specs):
        problems = []
        for spec in problem_specs:
            f = np.array(spec["front"])
            u = np.array(spec["up"])
            d = np.array(spec["down"])
            l = np.array(spec["left"])
            r = np.array(spec["right"])
            b = np.array(spec["back"])
            cube_state = Cube3State(f, u, d, l, r, b)
            problem = Problem(
                id=spec["id"],
                domain=Cube3(initial_state=cube_state),
            )
            problems.append(problem)
        return problems

    model_args = {
        "num_actions": problemset["num_actions"],
        "in_channels": problemset["in_channels"],
        "state_t_width": problemset["state_t_width"],
        "requires_backward_goal": True,
        "kernel_depth": 2,
        "state_t_depth": 6,
    }

    if "is_curriculum" in problemset:
        bootstrap_problems = parse_specs(problemset["bootstrap_problems"])
        problemset["bootstrap_problems"] = bootstrap_problems
        problemset["curriculum_problems"] = parse_specs(
            problemset["curriculum_problems"]
        )
        problemset["permutation_problems"] = parse_specs(
            problemset["permutation_problems"]
        )
    else:
        problems = parse_specs(problemset["problems"])
        problemset["problems"] = problems

    return problemset, model_args
