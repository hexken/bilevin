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
from typing import Optional

import numpy as np
import torch as to

from domains.domain import Domain, State, Problem
from enums import FourDir


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
    goal_str = "."
    man_str = "@"
    wall_str = "#"
    box_str = "$"

    wall_channel = 0
    goal_channel = 1
    box_channel = 2
    man_channel = 3

    def __init__(
        self,
        map: np.ndarray,
        man_row: int,
        man_col: int,
        boxes: np.ndarray,
        forward: bool = True,
    ) -> None:
        super().__init__()
        self.forward = forward

        self.map = map
        self.rows = map.shape[0]
        self.cols = map.shape[1]

        self.original_man_row = man_row
        self.original_man_col = man_col
        self.original_boxes = boxes
        self._initial_state = SokobanState(man_row, man_col, boxes)
        self.initial_state_t = self.state_tensor(self.initial_state)

    @property
    def initial_state(self) -> SokobanState:
        return self._initial_state

    @property
    def state_width(cls) -> int:
        return 10

    @property
    def num_actions(cls) -> int:
        return 4

    @property
    def in_channels(self) -> int:
        return 4

    def actions_unpruned(self, state: SokobanState) -> list[FourDir]:
        actions = []
        man_row = state.man_row
        man_col = state.man_col

        if (
            self.map[man_row, man_col + 1, Sokoban.wall_channel] == 0
            and state.boxes[man_row, man_col + 1] == 0
        ):
            actions.append(FourDir.RIGHT)
        elif (
            state.boxes[man_row, man_col + 1] == 1
            and self.map[man_row, man_col + 2, Sokoban.wall_channel] == 0
            and state.boxes[man_row, man_col + 2] == 0
        ):
            actions.append(FourDir.RIGHT)

        if (
            self.map[man_row, man_col - 1][Sokoban.wall_channel] == 0
            and state.boxes[man_row, man_col - 1] == 0
        ):
            actions.append(FourDir.LEFT)
        elif (
            state.boxes[man_row, man_col - 1] == 1
            and self.map[man_row, man_col - 2, Sokoban.wall_channel] == 0
            and state.boxes[man_row, man_col - 2] == 0
        ):
            actions.append(FourDir.LEFT)

        if (
            self.map[man_row + 1, man_col, Sokoban.wall_channel] == 0
            and state.boxes[man_row + 1, man_col] == 0
        ):
            actions.append(FourDir.DOWN)
        elif (
            state.boxes[man_row + 1, man_col] == 1
            and self.map[man_row + 2, man_col, Sokoban.wall_channel] == 0
            and state.boxes[man_row + 2, man_col] == 0
        ):
            actions.append(FourDir.DOWN)

        if (
            self.map[man_row - 1, man_col, Sokoban.wall_channel] == 0
            and state.boxes[man_row - 1, man_col] == 0
        ):

            actions.append(FourDir.UP)
        elif (
            state.boxes[man_row - 1, man_col] == 1
            and self.map[man_row - 2, man_col, Sokoban.wall_channel] == 0
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

    def reverse_action(self, action: FourDir) -> FourDir:
        if action == FourDir.UP:
            return FourDir.DOWN
        elif action == FourDir.DOWN:
            return FourDir.UP
        elif action == FourDir.LEFT:
            return FourDir.RIGHT
        elif action == FourDir.RIGHT:
            return FourDir.LEFT

    def _backward_actions_unpruned(self, state: SokobanState) -> list[FourDir]:
        assert not self.forward
        actions = []
        man_row = state.man_row
        man_col = state.man_col

        if self.map[man_row, man_col + 1, Sokoban.wall_channel] == 0:
            actions.append(FourDir.RIGHT)

        if self.map[man_row, man_col - 1][Sokoban.wall_channel] == 0:
            actions.append(FourDir.LEFT)

        if self.map[man_row + 1, man_col, Sokoban.wall_channel] == 0:
            actions.append(FourDir.DOWN)

        if self.map[man_row - 1, man_col, Sokoban.wall_channel] == 0:
            actions.append(FourDir.UP)

        return actions

    def _backward_actions(
        self, parent_action: FourDir, state: SokobanState
    ) -> list[FourDir]:
        return self._backward_actions_unpruned(state)

    def _backward_result(self, state: SokobanState, action: FourDir) -> SokobanState:

        boxes = np.array(state.boxes)
        man_row = state.man_row
        man_col = state.man_col

        if action == FourDir.UP:
            if boxes[man_row + 1, man_col] == 1:
                boxes[man_row + 1, man_col] = 0
                boxes[man_row - 1, man_col] = 1
            man_row -= 1

        if action == FourDir.DOWN:
            if boxes[man_row - 1, man_col] == 1:
                boxes[man_row - 1, man_col] = 0
                boxes[man_row + 1, man_col] = 1
            man_row += 1

        if action == FourDir.RIGHT:
            if boxes[man_row, man_col - 1] == 1:
                boxes[man_row, man_col - 1] = 0
                boxes[man_row, man_col + 1] = 1
            man_col += 1

        if action == FourDir.LEFT:
            if boxes[man_row, man_col + 1] == 1:
                boxes[man_row, man_col + 1] = 0
                boxes[man_row, man_col - 1] = 1
            man_col -= 1

        return SokobanState(man_row, man_col, boxes)

    def backward_domain(self) -> list[Sokoban]:
        assert self.forward
        boxes = np.argwhere(self.map[:, :, Sokoban.box_channel])
        map = np.array(self.map)
        domains = []
        for box in boxes:
            if not map[
                box[0] - 1, box[1], [Sokoban.wall_channel, Sokoban.box_channel]
            ].any():
                new_domain = Sokoban(map, box[0] - 1, box[1], boxes, forward=False)
                domains.append(new_domain)

            if not map[
                box[0] + 1, box[1], [Sokoban.wall_channel, Sokoban.box_channel]
            ].any():
                new_domain = Sokoban(map, box[0] + 1, box[1], boxes, forward=False)
                domains.append(new_domain)

            if not map[
                box[0], box[1] + 1, [Sokoban.wall_channel, Sokoban.box_channel]
            ].any():
                new_domain = Sokoban(map, box[0], box[1] + 1, boxes, forward=False)
                domains.append(new_domain)

            if not map[
                box[0], box[1] - 1, [Sokoban.wall_channel, Sokoban.box_channel]
            ].any():
                new_domain = Sokoban(map, box[0], box[1] - 1, boxes, forward=False)
                domains.append(new_domain)

        for new_domain in domains:
            new_domain.goal_state = self.initial_state
            new_domain.goal_state_t = self.initial_state_t
            new_domain.actions = self._backward_actions
            new_domain.actions_unpruned = self._backward_actions_unpruned
            new_domain.result = self._backward_result
            new_domain.is_goal = self._backward_is_goal

        return domains

    def _backward_is_goal(self, state: SokobanState) -> bool:
        assert not self.forward
        return state == self.goal_state

    def is_goal(self, state: SokobanState) -> bool:
        return self.map[:, :, Sokoban.goal_channel][
            state.boxes[:, 0], state.boxes[:, 1]
        ].all()

    def state_tensor(self, state: SokobanState) -> to.Tensor:
        channel_man = np.zeros((self.rows, self.cols))
        channel_man[state.man_row, state.man_col] = 1

        channel_boxes = np.zeros((self.rows, self.cols))
        channel_boxes[state.boxes[:, 0], state.boxes[:, 1]] = 1

        arr = np.concatenate(
            (self.map, channel_boxes[..., None], channel_man[..., None]), axis=-1
        )
        return to.from_numpy(arr)

    def print(self, state: SokobanState):
        boxes_list = state.boxes.tolist()
        for i in range(self.rows):
            for j in range(self.cols):
                loc = [i, j]
                if self.map[i, j, Sokoban.goal_channel] == 1 and loc in boxes_list:
                    print("*", end="")
                elif i == state.man_row and j == state.man_col:
                    print(Sokoban.man_str, end="")
                elif self.map[i, j, Sokoban.goal_channel] == 1:
                    print(Sokoban.goal_str, end="")
                elif self.map[i, j, Sokoban.wall_channel] == 1:
                    print(Sokoban.wall_str, end="")
                elif loc in boxes_list:
                    print(Sokoban.box_str, end="")
                else:
                    print(" ", end="")
            print()


def from_string(
    string_state: str,
) -> Sokoban:
    lines = string_state.splitlines()
    rows = len(lines)
    cols = len(lines[0])

    map = np.zeros((rows, cols, 2), dtype=np.float32)
    boxes = []
    man_row = -1
    man_col = -1

    for i in range(rows):
        for j in range(cols):
            if lines[i][j] == Sokoban.goal_str:
                map[i][j][Sokoban.goal_channel] = 1

            if lines[i][j] == Sokoban.man_str:
                man_row = i
                man_col = j

            if lines[i][j] == Sokoban.wall_str:
                map[i][j][Sokoban.wall_channel] = 1

            if lines[i][j] == Sokoban.box_str:
                boxes.append([i, j])

    boxes = np.array(boxes)

    return Sokoban(map, man_row, man_col, boxes)


def parse_problemset(problemset: dict):
    def parse_specs(problem_specs):
        problems = []
        for spec in problem_specs:
            problem = Problem(
                id=spec["id"],
                domain=Sokoban(
                    map=spec["map"],
                    man_row=spec["man_row"],
                    man_col=spec["man_col"],
                    boxes=spec["boxes"],
                ),
            )
            problems.append(problem)
        return problems

    model_args = {
        "num_actions": problemset["num_actions"],
        "in_channels": problemset["in_channels"],
        "state_t_width": problemset["state_t_width"],
        "width": problemset["width"],
    }

    problems = parse_specs(problemset["problems"])
    problemset["problems"] = problems

    return problemset, model_args
