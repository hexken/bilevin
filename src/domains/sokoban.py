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
    def __init__(self) -> None:
        super().__init__()

    def __eq__(self) -> bool:
        return False

    def __hash__(self) -> int:
        return 1


class Sokoban(Domain):
    def __init__(self, string_state=[]):

        self._channel_walls = 0
        self._channel_goals = 1

        self._channel_boxes = 2
        self._channel_man = 3

        self._goal = "."
        self._man = "@"
        self._wall = "#"
        self._box = "$"

        self._E = 0
        self._W = 1
        self._N = 2
        self._S = 3

        self._number_channels = 4

        if len(string_state) > 0:
            self._width = len(string_state[0])
            self._height = len(string_state)

            self._maze = np.zeros((self._height, self._width, 2))
            self._boxes = np.zeros((self._height, self._width))
            # self._puzzle = np.zeros((self._height, self._width, self._number_channels))

            for i in range(self._height):
                for j in range(self._width):
                    if string_state[i][j] == self._goal:
                        self._maze[i][j][self._channel_goals] = 1

                    if string_state[i][j] == self._man:
                        self._y_man = i
                        self._x_man = j

                    if string_state[i][j] == self._wall:
                        self._maze[i][j][self._channel_walls] = 1

                    if string_state[i][j] == self._box:
                        self._boxes[i][j] = 1

    def copy(self):
        copy_state = Sokoban()

        copy_state._width = self._width
        copy_state._height = self._height
        copy_state._maze = self._maze

        copy_state._x_man = self._x_man
        copy_state._y_man = self._y_man

        copy_state._boxes = self._boxes.copy()

        return copy_state

    def __hash__(self):
        return hash((str(self._boxes), str(self._man)))

    def __eq__(self, other):
        return (
            np.array_equal(self._boxes, other._boxes)
            and self._x_man == other._x_man
            and self._y_man == other._y_man
        )

    def successors(self):
        actions = []

        if self._x_man + 1 < self._width:
            if (
                self._maze[self._y_man][self._x_man + 1][self._channel_walls] == 0
                and self._boxes[self._y_man][self._x_man + 1] == 0
            ):

                actions.append(self._E)
            elif (
                self._maze[self._y_man][self._x_man + 1][self._channel_walls] == 0
                and self._boxes[self._y_man][self._x_man + 1] == 1
                and self._x_man + 2 < self._width
                and self._maze[self._y_man][self._x_man + 2][self._channel_walls] == 0
                and self._boxes[self._y_man][self._x_man + 2] == 0
            ):

                actions.append(self._E)

        if self._x_man - 1 > 0:
            if (
                self._maze[self._y_man][self._x_man - 1][self._channel_walls] == 0
                and self._boxes[self._y_man][self._x_man - 1] == 0
            ):

                actions.append(self._W)
            elif (
                self._maze[self._y_man][self._x_man - 1][self._channel_walls] == 0
                and self._boxes[self._y_man][self._x_man - 1] == 1
                and self._x_man - 2 > 0
                and self._maze[self._y_man][self._x_man - 2][self._channel_walls] == 0
                and self._boxes[self._y_man][self._x_man - 2] == 0
            ):

                actions.append(self._W)

        if self._y_man + 1 < self._height:
            if (
                self._maze[self._y_man + 1][self._x_man][self._channel_walls] == 0
                and self._boxes[self._y_man + 1][self._x_man] == 0
            ):

                actions.append(self._S)
            elif (
                self._maze[self._y_man + 1][self._x_man][self._channel_walls] == 0
                and self._boxes[self._y_man + 1][self._x_man] == 1
                and self._y_man + 2 < self._height
                and self._maze[self._y_man + 2][self._x_man][self._channel_walls] == 0
                and self._boxes[self._y_man + 2][self._x_man] == 0
            ):

                actions.append(self._S)

        if self._y_man - 1 > 0:
            if (
                self._maze[self._y_man - 1][self._x_man][self._channel_walls] == 0
                and self._boxes[self._y_man - 1][self._x_man] == 0
            ):

                actions.append(self._N)
            elif (
                self._maze[self._y_man - 1][self._x_man][self._channel_walls] == 0
                and self._boxes[self._y_man - 1][self._x_man] == 1
                and self._y_man - 2 > 0
                and self._maze[self._y_man - 2][self._x_man][self._channel_walls] == 0
                and self._boxes[self._y_man - 2][self._x_man] == 0
            ):

                actions.append(self._N)

        return actions

    def successors_parent_pruning(self, op):
        actions = []

        return self.successors()

        #         if self._x_man + 1 < self._width and not (op == self._W and self._puzzle[self._y_man][self._x_man - 1][self._channel_boxes] == 0):
        #             if (self._puzzle[self._y_man][self._x_man + 1][self._channel_walls] == 0 and
        #                 self._puzzle[self._y_man][self._x_man + 1][self._channel_boxes] == 0):
        #
        #                 actions.append(self._E)
        #             elif (self._puzzle[self._y_man][self._x_man + 1][self._channel_walls] == 0 and
        #                 self._puzzle[self._y_man][self._x_man + 1][self._channel_boxes] == 1 and
        #                 self._x_man + 2 < self._width and
        #                 self._puzzle[self._y_man][self._x_man + 2][self._channel_walls] == 0 and
        #                 self._puzzle[self._y_man][self._x_man + 2][self._channel_boxes] == 0):
        #
        #                 actions.append(self._E)
        #
        #         if self._x_man - 1 > 0 and not (op == self._E and self._puzzle[self._y_man][self._x_man + 1][self._channel_boxes] == 0):
        #             if (self._puzzle[self._y_man][self._x_man - 1][self._channel_walls] == 0 and
        #                 self._puzzle[self._y_man][self._x_man - 1][self._channel_boxes] == 0):
        #
        #                 actions.append(self._W)
        #             elif (self._puzzle[self._y_man][self._x_man - 1][self._channel_walls] == 0 and
        #                 self._puzzle[self._y_man][self._x_man - 1][self._channel_boxes] == 1 and
        #                 self._x_man - 2 > 0 and
        #                 self._puzzle[self._y_man][self._x_man - 2][self._channel_walls] == 0 and
        #                 self._puzzle[self._y_man][self._x_man - 2][self._channel_boxes] == 0):
        #
        #                 actions.append(self._W)
        #
        #         if self._y_man + 1 < self._height and not (op == self._N and self._puzzle[self._y_man - 1][self._x_man][self._channel_boxes] == 0):
        #             if (self._puzzle[self._y_man + 1][self._x_man][self._channel_walls] == 0 and
        #                 self._puzzle[self._y_man + 1][self._x_man][self._channel_boxes] == 0):
        #
        #                 actions.append(self._S)
        #             elif (self._puzzle[self._y_man + 1][self._x_man][self._channel_walls] == 0 and
        #                 self._puzzle[self._y_man + 1][self._x_man][self._channel_boxes] == 1 and
        #                 self._y_man + 2 < self._height and
        #                 self._puzzle[self._y_man + 2][self._x_man][self._channel_walls] == 0 and
        #                 self._puzzle[self._y_man + 2][self._x_man][self._channel_boxes] == 0):
        #
        #                 actions.append(self._S)
        #
        #         if self._y_man - 1 > 0 and not (op == self._S and self._puzzle[self._y_man + 1][self._x_man][self._channel_boxes] == 0):
        #             if (self._puzzle[self._y_man - 1][self._x_man][self._channel_walls] == 0 and
        #                 self._puzzle[self._y_man - 1][self._x_man][self._channel_boxes] == 0):
        #
        #                 actions.append(self._N)
        #             elif (self._puzzle[self._y_man - 1][self._x_man][self._channel_walls] == 0 and
        #                 self._puzzle[self._y_man - 1][self._x_man][self._channel_boxes] == 1 and
        #                 self._y_man - 2 > 0 and
        #                 self._puzzle[self._y_man - 2][self._x_man][self._channel_walls] == 0 and
        #                 self._puzzle[self._y_man - 2][self._x_man][self._channel_boxes] == 0):
        #
        #                 actions.append(self._N)

        return actions

    def apply_action(self, action):

        if action == self._N:
            if self._boxes[self._y_man - 1][self._x_man] == 1:
                self._boxes[self._y_man - 1][self._x_man] = 0
                self._boxes[self._y_man - 2][self._x_man] = 1
            # self._puzzle[self._y_man][self._x_man][self._channel_man] = 0
            # self._puzzle[self._y_man - 1][self._x_man][self._channel_man] = 1
            self._y_man = self._y_man - 1

        if action == self._S:
            if self._boxes[self._y_man + 1][self._x_man] == 1:
                self._boxes[self._y_man + 1][self._x_man] = 0
                self._boxes[self._y_man + 2][self._x_man] = 1
            # self._puzzle[self._y_man][self._x_man][self._channel_man] = 0
            # self._puzzle[self._y_man + 1][self._x_man][self._channel_man] = 1
            self._y_man = self._y_man + 1

        if action == self._E:
            if self._boxes[self._y_man][self._x_man + 1] == 1:
                self._boxes[self._y_man][self._x_man + 1] = 0
                self._boxes[self._y_man][self._x_man + 2] = 1
            # self._puzzle[self._y_man][self._x_man][self._channel_man] = 0
            # self._puzzle[self._y_man][self._x_man + 1][self._channel_man] = 1
            self._x_man = self._x_man + 1

        if action == self._W:
            if self._boxes[self._y_man][self._x_man - 1] == 1:
                self._boxes[self._y_man][self._x_man - 1] = 0
                self._boxes[self._y_man][self._x_man - 2] = 1
            #             self._puzzle[self._y_man][self._x_man][self._channel_man] = 0
            #             self._puzzle[self._y_man][self._x_man - 1][self._channel_man] = 1
            self._x_man = self._x_man - 1

    def is_solution(self):
        for i in range(self._height):
            for j in range(self._width):
                if (
                    self._boxes[i][j] == 1
                    and self._maze[i][j][self._channel_goals] == 0
                ):
                    return False
        return True

    def get_image_representation(self):
        image = np.zeros((self._height, self._width, self._number_channels))
        for i in range(self._height):
            for j in range(self._width):
                image[i][j][self._channel_goals] = self._maze[i][j][self._channel_goals]
                image[i][j][self._channel_walls] = self._maze[i][j][self._channel_walls]
                image[i][j][self._channel_boxes] = self._boxes[i][j]

        image[self._y_man][self._x_man][self._channel_man] = 1

        return image

    def heuristic_value(self):
        h = 0
        h_man = self._width + self._height

        for i in range(self._height):
            for j in range(self._width):
                if (
                    self._boxes[i][j] == 1
                    and self._maze[i][j][self._channel_goals] == 0
                ):
                    h_box = self._width + self._height
                    for l in range(self._height):
                        for m in range(self._width):
                            if self._maze[l][m][self._channel_goals] == 1:
                                dist_to_goal = abs(l - i) + abs(m - j)
                                if dist_to_goal < h_box:
                                    h_box = dist_to_goal
                    h += h_box

                if self._boxes[i][j] == 1:
                    dist_to_man = abs(self._y_man - i) + abs(self._x_man - j) - 1
                    if dist_to_man < h_man:
                        h_man = dist_to_man
        h += h_man
        return h

    def print(self):
        for i in range(self._height):
            for j in range(self._width):
                if (
                    self._maze[i][j][self._channel_goals] == 1
                    and self._boxes[i][j] == 1
                ):
                    print("*", end="")
                elif i == self._y_man and j == self._x_man:
                    print(self._man, end="")
                elif self._maze[i][j][self._channel_goals] == 1:
                    print(self._goal, end="")
                elif self._maze[i][j][self._channel_walls] == 1:
                    print(self._wall, end="")
                elif self._boxes[i][j] == 1:
                    print(self._box, end="")
                else:
                    print(" ", end="")
            print()

