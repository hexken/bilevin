# Copyright (C) 2021-2022, Ken Tjhia
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations
from collections import deque
from copy import deepcopy
from typing import Optional, Type

import matplotlib.pyplot as plt
import numpy as np
import torch as to

from domains.domain import Domain, State
from enums import Color, FourDir
from search.utils import SearchNode, Trajectory


class SokobanState(State):
    def __init__(self, man_row: int, man_col: int, boxes: np.ndarray) -> None:
        self.man_row = man_row
        self.man_col = man_col
        self.boxes = boxes

    def __eq__(self, other) -> bool:
        return (
            self.man_row == other.man_row
            and self.man_col == other.man_col
            and np.array_equal(self.boxes, other.boxes)
        )

    def __hash__(self) -> int:
        return (self.man_row, self.man_col, self.boxes.tobytes()).__hash__()


class Sokoban(Domain):
    def __init__(
        self, map: np.ndarray, man_row: int, man_col: int, boxes: np.ndarray
    ) -> None:

        self._channel_walls = 0
        self._channel_goals = 1
        self._channel_boxes = 2
        self._channel_man = 3
        self._number_channels = 4

        self._goal = "."
        self._man = "@"
        self._wall = "#"
        self._box = "$"

        self.map = map
        self.rows = map.shape[1]
        self.cols = map.shape[0]

        self.initial_state = SokobanState(man_row, man_col, boxes)
        # todo state tensors, backward state

    def _parse_string(
        self, string_state: str
    ) -> Optional[tuple[np.ndarray, int, int, np.ndarray]]:
        if len(string_state) > 0:
            cols = len(string_state[0])
            rows = len(string_state)

            map = np.zeros((rows, cols, 2))
            boxes = np.zeros((rows, cols))
            man_row = -1
            man_col = -1

            for i in range(rows):
                for j in range(cols):
                    if string_state[i][j] == self._goal:
                        map[i][j][self._channel_goals] = 1

                    if string_state[i][j] == self._man:
                        man_row = i
                        man_col = j

                    if string_state[i][j] == self._wall:
                        map[i][j][self._channel_walls] = 1

                    if string_state[i][j] == self._box:
                        boxes[i, j] = 1

            boxes = np.array(boxes)
            return map, man_row, man_col, boxes

    @property
    def num_actions(cls) -> int:
        return 4

    @property
    def in_channels(self) -> int:
        return self._number_channels

    def actions_unpruned(self, state: SokobanState) -> list[FourDir]:
        actions = []
        man_row = state.man_row
        man_col = state.man_col

        if (
            self.map[man_row, man_col + 1, self._channel_walls] == 0
            and state.boxes[man_row, man_col + 1] == 0
        ):
            actions.append(FourDir.RIGHT)
        elif (
            state.boxes[man_row, man_col + 1] == 1
            and self.map[man_row, man_col + 2, self._channel_walls] == 0
            and state.boxes[man_row, man_col + 2] == 0
        ):
            actions.append(FourDir.RIGHT)

        if (
            self.map[man_row, man_col - 1][self._channel_walls] == 0
            and state.boxes[man_row, man_col - 1] == 0
        ):
            actions.append(FourDir.LEFT)
        elif (
            state.boxes[man_row, man_col - 1] == 1
            and self.map[man_row, man_col - 2, self._channel_walls] == 0
            and state.boxes[man_row, man_col - 2] == 0
        ):
            actions.append(FourDir.LEFT)

        if (
            self.map[man_row + 1, man_col, self._channel_walls] == 0
            and state.boxes[man_row + 1, man_col] == 0
        ):
            actions.append(FourDir.DOWN)
        elif (
            state.boxes[man_row + 1, man_col] == 1
            and self.map[man_row + 2, man_col, self._channel_walls] == 0
            and state.boxes[man_row + 2, man_col] == 0
        ):
            actions.append(FourDir.DOWN)

        if (
            self.map[man_row - 1, man_col, self._channel_walls] == 0
            and state.boxes[man_row - 1, man_col] == 0
        ):

            actions.append(FourDir.UP)
        elif (
            state.boxes[man_row - 1, man_col] == 1
            and self.map[man_row - 2, man_col, self._channel_walls] == 0
            and state.boxes[man_row - 2, man_col] == 0
        ):
            actions.append(FourDir.UP)

        return actions

    def actions(self, parent_action: FourDir, state: SokobanState) -> list[FourDir]:
        return self.actions_unpruned(state)

    def result(self, state: SokobanState, action: FourDir) -> SokobanState:

        boxes = np.array(state.boxes)
        man_row = state.man_row
        man_col = state.man_col

        if action == FourDir.UP:
            if boxes[man_row - 1, man_col] == 1:
                boxes[man_row - 1, man_col] = 0
                boxes[man_row - 2, man_col] = 1
            man_row -= 1

        if action == FourDir.DOWN:
            if boxes[man_row + 1, man_col] == 1:
                boxes[man_row + 1, man_col] = 0
                boxes[man_row + 2, man_col] = 1
            man_row += 1

        if action == FourDir.RIGHT:
            if boxes[man_row, man_col + 1] == 1:
                boxes[man_row, man_col + 1] = 0
                boxes[man_row, man_col + 2] = 1
            man_col += 1

        if action == FourDir.LEFT:
            if boxes[man_row, man_col - 1] == 1:
                boxes[man_row, man_col - 1] = 0
                boxes[man_row, man_col - 2] = 1
            man_col -= 1

        return SokobanState(man_row, man_col, boxes)

    def is_goal(self, state: SokobanState) -> bool:
        return self.map[:, :, self._channel_goals][
            state.boxes[:, 0], state.boxes[:, 1]
        ].all()

    def state_tensor(self, state: SokobanState) -> to.Tensor:
        channel_man = np.zeros((self.cols, self.rows))
        channel_man[state.man_row, state.man_col] = 1
        arr = np.concatenate((self.map, channel_man), axis=-1)

        return to.from_numpy(arr)

    def print(self, state: SokobanState):
        boxes_list = state.boxes.tolist()
        for i in range(self.rows):
            for j in range(self.cols):
                loc = [i, j]
                if self.map[i, j, self._channel_goals] == 1 and loc in boxes_list:
                    print("*", end="")
                elif i == state.man_row and j == state.man_col:
                    print(self._man, end="")
                elif self.map[i, j, self._channel_goals] == 1:
                    print(self._goal, end="")
                elif self.map[i, j, self._channel_walls] == 1:
                    print(self._wall, end="")
                elif loc in boxes_list:
                    print(self._box, end="")
                else:
                    print(" ", end="")
            print()
