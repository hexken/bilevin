"""
Based on the implementation found at this nice blog post:
    https://blog.dorianbolivar.com/2021/01/solving-rubik-cycle-programming.html
    https://github.com/dbolivar/programming_challenges_exercises/blob/main/URI_Online_Judge/1232-Rubik_Cycle/1232_numpy.py
"""
from __future__ import annotations

from numba import jit, njit
import numpy as np
import torch as to
from torch import from_numpy
from torch.nn.functional import one_hot

from domains.domain import Domain, State
from search.utils import Problem


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
        self._forward_result = self.result

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
        domain.result = domain._backwards_result
        domain.goal_state = self.initial_state
        domain.goal_state_t = self.state_tensor(self.initial_state)
        return domain

    def is_goal2(self, state: Cube3State) -> bool:
        return goal_check(
            state.front, state.up, state.down, state.left, state.right, state.back
        )

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
                np.flip(l[:, 2]),
                d[0, :],
                np.flip(r[:, 0]),
                u[2, :],
            )
        elif action == 1:
            f = np.rot90(f, 1)
            u[2, :], r[:, 0], d[0, :], l[:, 2] = (
                r[:, 0],
                np.flip(d[0, :]),
                l[:, 2],
                np.flip(u[2, :]),
            )
        elif action == 2:
            # Face rotate.
            u = np.rot90(u, -1)
            b[2, :], l[0, :], f[0, :], r[0, :] = (
                np.flip(l[0, :]),
                f[0, :],
                r[0, :],
                np.flip(b[2, :]),
            )
        elif action == 3:
            u = np.rot90(u, 1)
            b[2, :], r[0, :], f[0, :], l[0, :] = (
                np.flip(r[0, :]),
                f[0, :],
                l[0, :],
                np.flip(b[2, :]),
            )
        elif action == 4:
            d = np.rot90(d, -1)
            f[2, :], l[2, :], b[0, :], r[2, :] = (
                l[2, :],
                np.flip(b[0, :]),
                np.flip(r[2, :]),
                f[2, :],
            )
        elif action == 5:
            d = np.rot90(d, 1)
            f[2, :], r[2, :], b[0, :], l[2, :] = (
                r[2, :],
                np.flip(b[0, :]),
                np.flip(l[2, :]),
                f[2, :],
            )
        elif action == 6:
            l = np.rot90(l, -1)
            u[:, 0], b[:, 0], d[:, 0], f[:, 0] = (
                b[:, 0],
                d[:, 0],
                f[:, 0],
                u[:, 0],
            )
        elif action == 7:
            l = np.rot90(l, 1)
            u[:, 0], f[:, 0], d[:, 0], b[:, 0] = (
                f[:, 0],
                d[:, 0],
                b[:, 0],
                u[:, 0],
            )
        elif action == 8:
            r = np.rot90(r, -1)
            u[:, 2], f[:, 2], d[:, 2], b[:, 2] = (
                f[:, 2],
                d[:, 2],
                b[:, 2],
                u[:, 2],
            )
        elif action == 9:
            r = np.rot90(r, 1)
            u[:, 2], b[:, 2], d[:, 2], f[:, 2] = (
                b[:, 2],
                d[:, 2],
                f[:, 2],
                u[:, 2],
            )
        elif action == 10:
            b = np.rot90(b, -10)
            u[0, :], r[:, 2], d[2, :], l[:, 0] = (
                r[:, 2],
                np.flip(d[2, :]),
                l[:, 0],
                np.flip(u[0, :]),
            )
        else:
            b = np.rot90(b, -1)
            u[0, :], l[:, 0], d[2, :], r[:, 2] = (
                np.flip(l[:, 0]),
                d[2, :],
                np.flip(r[:, 2]),
                u[0, :],
            )

        return Cube3State(f, u, d, l, r, b)

    def result2(self, state: Cube3State, action: int) -> Cube3State:
        return Cube3State(
            *faces_result(
                state.front,
                state.up,
                state.down,
                state.left,
                state.right,
                state.back,
                action,
            )
        )

    def _backwards_result(self, state: Cube3State, action: int) -> Cube3State:
        return self._forward_result(state, self.reverse_action(action))

    def is_goal(self, state: Cube3State) -> bool:
        return (
            (state.front == 0).all()
            and (state.up == 1).all()
            and (state.down == 2).all()
            and (state.left == 3).all()
            and (state.right == 4).all()
            and (state.back == 5).all()
        )
        # for i in range(3):
        #     for j in range(3):
        #         if (
        #             state.front[i, j] != 0
        #             or state.up[i, j] != 1
        #             or state.down[i, j] != 2
        #             or state.left[i, j] != 3
        #             or state.right[i, j] != 4
        #             or state.back[i, j] != 5
        #         ):
        #             return False
        # return True

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


# @njit
# def faces_result(
#     front: np.ndarray,
#     up: np.ndarray,
#     down: np.ndarray,
#     left: np.ndarray,
#     right: np.ndarray,
#     back: np.ndarray,
#     action: int,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     f = front.copy()
#     u = up.copy()
#     d = down.copy()
#     l = left.copy()
#     r = right.copy()
#     b = back.copy()

#     if action == 0:
#         f = np.rot90(f, -1)
#         u[2, :], l[:, 2], d[0, :], r[:, 0] = (
#             np.flip(l[:, 2]),
#             d[0, :],
#             np.flip(r[:, 0]),
#             u[2, :],
#         )
#     elif action == 1:
#         f = np.rot90(f, 1)
#         u[2, :], r[:, 0], d[0, :], l[:, 2] = (
#             r[:, 0],
#             np.flip(d[0, :]),
#             l[:, 2],
#             np.flip(u[2, :]),
#         )
#     elif action == 2:
#         # Face rotate.
#         u = np.rot90(u, -1)
#         b[2, :], l[0, :], f[0, :], r[0, :] = (
#             np.flip(l[0, :]),
#             f[0, :],
#             r[0, :],
#             np.flip(b[2, :]),
#         )
#     elif action == 3:
#         u = np.rot90(u, 1)
#         b[2, :], r[0, :], f[0, :], l[0, :] = (
#             np.flip(r[0, :]),
#             f[0, :],
#             l[0, :],
#             np.flip(b[2, :]),
#         )
#     elif action == 4:
#         d = np.rot90(d, -1)
#         f[2, :], l[2, :], b[0, :], r[2, :] = (
#             l[2, :],
#             np.flip(b[0, :]),
#             np.flip(r[2, :]),
#             f[2, :],
#         )
#     elif action == 5:
#         d = np.rot90(d, 1)
#         f[2, :], r[2, :], b[0, :], l[2, :] = (
#             r[2, :],
#             np.flip(b[0, :]),
#             np.flip(l[2, :]),
#             f[2, :],
#         )
#     elif action == 6:
#         l = np.rot90(l, -1)
#         u[:, 0], b[:, 0], d[:, 0], f[:, 0] = (
#             b[:, 0],
#             d[:, 0],
#             f[:, 0],
#             u[:, 0],
#         )
#     elif action == 7:
#         l = np.rot90(l, 1)
#         u[:, 0], f[:, 0], d[:, 0], b[:, 0] = (
#             f[:, 0],
#             d[:, 0],
#             b[:, 0],
#             u[:, 0],
#         )
#     elif action == 8:
#         r = np.rot90(r, -1)
#         u[:, 2], f[:, 2], d[:, 2], b[:, 2] = (
#             f[:, 2],
#             d[:, 2],
#             b[:, 2],
#             u[:, 2],
#         )
#     elif action == 9:
#         r = np.rot90(r, 1)
#         u[:, 2], b[:, 2], d[:, 2], f[:, 2] = (
#             b[:, 2],
#             d[:, 2],
#             f[:, 2],
#             u[:, 2],
#         )
#     elif action == 10:
#         b = np.rot90(b, -10)
#         u[0, :], r[:, 2], d[2, :], l[:, 0] = (
#             r[:, 2],
#             np.flip(d[2, :]),
#             l[:, 0],
#             np.flip(u[0, :]),
#         )
#     else:
#         b = np.rot90(b, -1)
#         u[0, :], l[:, 0], d[2, :], r[:, 2] = (
#             np.flip(l[:, 0]),
#             d[2, :],
#             np.flip(r[:, 2]),
#             u[0, :],
#         )

#     return f, u, d, l, r, b


def get_goal_state() -> Cube3State:
    front = np.full((3, 3), 0, dtype=int)
    up = np.full((3, 3), 1, dtype=int)
    down = np.full((3, 3), 2, dtype=int)
    left = np.full((3, 3), 3, dtype=int)
    right = np.full((3, 3), 4, dtype=int)
    back = np.full((3, 3), 5, dtype=int)
    state = Cube3State(front, up, down, left, right, back)
    return state


# @njit
def goal_check(
    front: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    back: np.ndarray,
) -> bool:
    for i in range(3):
        for j in range(3):
            if (
                front[i, j] != 0
                or up[i, j] != 1
                or down[i, j] != 2
                or left[i, j] != 3
                or right[i, j] != 4
                or back[i, j] != 5
            ):
                return False
    return True


# @njit
def equal_check(
    front: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    back: np.ndarray,
    front2: np.ndarray,
    up2: np.ndarray,
    down2: np.ndarray,
    left2: np.ndarray,
    right2: np.ndarray,
    back2: np.ndarray,
) -> bool:
    for i in range(3):
        for j in range(3):
            if (
                front[i, j] != front2[i, j]
                or up[i, j] != up2[i, j]
                or down[i, j] != down2[i, j]
                or left[i, j] != left2[i, j]
                or right[i, j] != right2[i, j]
                or back[i, j] != back2[i, j]
            ):
                return False
    return True


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
