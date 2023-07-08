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

    def __eq__(self, other):
        return np.array_equal(self.colors, other.colors)


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


def show(faces: str = "f,u,d,l,r,b"):
    for face in faces.split(","):
        print(f"{face.strip()}:\n{eval(face)}")


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
    # Uppercase: clockwise. Lowercase: counterclockwise.
    f = front.copy()
    u = up.copy()
    d = down.copy()
    l = left.copy()
    r = right.copy()
    b = back.copy()

    if move == 0:
        f = np.rot90(f, -1)
    elif move == 1:
        f = np.rot90(f, 1)
        # Edges rotation.
        if move.isupper():
            u[2, :], l[:, 2], d[0, :], r[:, 0] = (
                np.flip(l[:, 2]).copy(),
                d[0, :].copy(),
                np.flip(r[:, 0]).copy(),
                u[2, :].copy(),
            )
        else:
            u[2, :], r[:, 0], d[0, :], l[:, 2] = (
                r[:, 0].copy(),
                np.flip(d[0, :]).copy(),
                l[:, 2].copy(),
                np.flip(u[2, :]).copy(),
            )
    elif move in "Uu":
        # Face rotate.
        u = np.rot90(u, -1 if move.isupper() else 1)

        # Edges rotation.
        if move.isupper():
            b[2, :], l[0, :], f[0, :], r[0, :] = (
                np.flip(l[0, :]).copy(),
                f[0, :].copy(),
                r[0, :].copy(),
                np.flip(b[2, :]).copy(),
            )
        else:
            b[2, :], r[0, :], f[0, :], l[0, :] = (
                np.flip(r[0, :]).copy(),
                f[0, :].copy(),
                l[0, :].copy(),
                np.flip(b[2, :]).copy(),
            )
    elif move in "Dd":
        # Face rotate.
        d = np.rot90(d, -1 if move.isupper() else 1)

        # Edges rotation.
        if move.isupper():
            f[2, :], l[2, :], b[0, :], r[2, :] = (
                l[2, :].copy(),
                np.flip(b[0, :]).copy(),
                np.flip(r[2, :]).copy(),
                f[2, :].copy(),
            )
        else:
            f[2, :], r[2, :], b[0, :], l[2, :] = (
                r[2, :].copy(),
                np.flip(b[0, :]).copy(),
                np.flip(l[2, :]).copy(),
                f[2, :].copy(),
            )
    elif move in "Ll":
        # Face rotate.
        l = np.rot90(l, -1 if move.isupper() else 1)

        # Edges rotation.
        if move.isupper():
            u[:, 0], b[:, 0], d[:, 0], f[:, 0] = (
                b[:, 0].copy(),
                d[:, 0].copy(),
                f[:, 0].copy(),
                u[:, 0].copy(),
            )
        else:
            u[:, 0], f[:, 0], d[:, 0], b[:, 0] = (
                f[:, 0].copy(),
                d[:, 0].copy(),
                b[:, 0].copy(),
                u[:, 0].copy(),
            )
    elif move in "Rr":
        # Face rotate.
        r = np.rot90(r, -1 if move.isupper() else 1)

        # Edges rotation.
        if move.isupper():
            u[:, 2], f[:, 2], d[:, 2], b[:, 2] = (
                f[:, 2].copy(),
                d[:, 2].copy(),
                b[:, 2].copy(),
                u[:, 2].copy(),
            )
        else:
            u[:, 2], b[:, 2], d[:, 2], f[:, 2] = (
                b[:, 2].copy(),
                d[:, 2].copy(),
                f[:, 2].copy(),
                u[:, 2].copy(),
            )
    elif move in "Bb":
        # Face rotate.
        b = np.rot90(b, -1 if move.isupper() else 1)

        # Edges rotation.
        if move.isupper():
            u[0, :], r[:, 2], d[2, :], l[:, 0] = (
                r[:, 2].copy(),
                np.flip(d[2, :]).copy(),
                l[:, 0].copy(),
                np.flip(u[0, :]).copy(),
            )
        else:
            u[0, :], l[:, 0], d[2, :], r[:, 2] = (
                np.flip(l[:, 0]).copy(),
                d[2, :].copy(),
                np.flip(r[:, 2]).copy(),
                u[0, :].copy(),
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
