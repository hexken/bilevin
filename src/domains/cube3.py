"""
Adapted from this nice blog post:
    https://blog.dorianbolivar.com/2021/01/solving-rubik-cycle-programming.html
    https://github.com/dbolivar/programming_challenges_exercises/blob/main/URI_Online_Judge/1232-Rubik_Cycle/1232_numpy.py
"""
from __future__ import annotations

from numba import jit, njit
import numpy as np
import torch as to

from domains import Domain, State


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
        return ""

    def __hash__(self) -> int:
        return (
            self.front.tobytes().__hash__(),
            self.up.tobytes().__hash__(),
            self.down.tobytes().__hash__(),
            self.left.tobytes().__hash__(),
            self.right.tobytes().__hash__(),
            self.back.tobytes().__hash__(),
        ).__hash__()

    def __eq__(self, other: Cube3State) -> bool:
        pass


class Cube3(Domain):
    def __init__(self, forward=True):
        super().__init__(forward=forward)
        self.cube_width: int = 3
        self._actions_list: list[int] = [i for i in range(self.num_actions)]

    @property
    def state_width(self) -> int:
        return 3

    @property
    def num_actions(cls) -> int:
        return 12

    @property
    def in_channels(self) -> int:
        return 6

    def set_to_goal(self, state: Cube3State) -> None:
        state.front = np.full((3, 3), 0, dtype=int)
        state.up = np.full((3, 3), 1, dtype=int)
        state.down = np.full((3, 3), 2, dtype=int)
        state.left = np.full((3, 3), 3, dtype=int)
        state.right = np.full((3, 3), 4, dtype=int)
        state.back = np.full((3, 3), 5, dtype=int)

    def _actions(self) -> list[int]:
        """
        0 FRONT GREEN
        1 UP WHITE
        2 DOWN YELLOW
        3 LEFT ORANGE
        4 RIGHT RED
        5 BACK BLUE
        """
        return self._actions_list

    def _actions_unpruned(self) -> list[int]:
        return self._actions_list

    @njit
    def reverse_action(self, action: int) -> int:
        if action & 1 == 1:  # odd
            return action - 1
        else:
            return action + 1

    def get_backward_domain(self) -> Cube3:
        pass

    def is_goal(self, state: Cube3State) -> bool:
        return face_check(
            state.front, state.up, state.down, state.left, state.right, state.back
        )

    # def is_goal(self, state: Cube3State) -> bool:
    #     for i in range(3):
    #         for j in range(3):
    #             if (
    #                 state.front[i, j] != 0
    #                 or state.up[i, j] != 1
    #                 or state.down[i, j] != 2
    #                 or state.left[i, j] != 3
    #                 or state.right[i, j] != 4
    #                 or state.back[i, j] != 5
    #             ):
    #                 return False
    #     return True

    def state_tensor(self, state: Cube3State) -> to.Tensor:
        pass


@njit
def faces_result(
    move: int,
    front: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    back: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f = front.copy()
    u = up.copy()
    d = down.copy()
    l = left.copy()
    r = right.copy()
    b = back.copy()

    if move == 0:
        f = np.rot90(f, -1)
        u[2, :], l[:, 2], d[0, :], r[:, 0] = (
            np.flip(l[:, 2]),
            d[0, :],
            np.flip(r[:, 0]),
            u[2, :],
        )
    elif move == 1:
        f = np.rot90(f, 1)
        u[2, :], r[:, 0], d[0, :], l[:, 2] = (
            r[:, 0],
            np.flip(d[0, :]),
            l[:, 2],
            np.flip(u[2, :]),
        )
    elif move == 2:
        # Face rotate.
        u = np.rot90(u, -1)
        b[2, :], l[0, :], f[0, :], r[0, :] = (
            np.flip(l[0, :]),
            f[0, :],
            r[0, :],
            np.flip(b[2, :]),
        )
    elif move == 3:
        u = np.rot90(u, 1)
        b[2, :], r[0, :], f[0, :], l[0, :] = (
            np.flip(r[0, :]),
            f[0, :],
            l[0, :],
            np.flip(b[2, :]),
        )
    elif move == 4:
        d = np.rot90(d, -1)
        f[2, :], l[2, :], b[0, :], r[2, :] = (
            l[2, :],
            np.flip(b[0, :]),
            np.flip(r[2, :]),
            f[2, :],
        )
    elif move == 5:
        d = np.rot90(d, 1)
        f[2, :], r[2, :], b[0, :], l[2, :] = (
            r[2, :],
            np.flip(b[0, :]),
            np.flip(l[2, :]),
            f[2, :],
        )
    elif move == 6:
        l = np.rot90(l, -1)
        u[:, 0], b[:, 0], d[:, 0], f[:, 0] = (
            b[:, 0],
            d[:, 0],
            f[:, 0],
            u[:, 0],
        )
    elif move == 7:
        l = np.rot90(l, 1)
        u[:, 0], f[:, 0], d[:, 0], b[:, 0] = (
            f[:, 0],
            d[:, 0],
            b[:, 0],
            u[:, 0],
        )
    elif move == 8:
        r = np.rot90(r, -1)
        u[:, 2], f[:, 2], d[:, 2], b[:, 2] = (
            f[:, 2],
            d[:, 2],
            b[:, 2],
            u[:, 2],
        )
    elif move == 9:
        r = np.rot90(r, 1)
        u[:, 2], b[:, 2], d[:, 2], f[:, 2] = (
            b[:, 2],
            d[:, 2],
            f[:, 2],
            u[:, 2],
        )
    elif move == 10:
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

    return f, u, d, l, r, b


@njit
def face_check(
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
