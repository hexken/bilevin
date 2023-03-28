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

from domains.domain import Domain, State
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
    def __init__(
        self,
        map: np.ndarray,
        man_row: int,
        man_col: int,
        boxes: np.ndarray,
        forward: bool = True,
    ) -> None:
        super().__init__()
        self._channel_walls = 0
        self._channel_goals = 1
        self._channel_boxes = 2
        self._channel_man = 3
        self._number_channels = 4
        self.forward = forward

        self._goal = "."
        self._man = "@"
        self._wall = "#"
        self._box = "$"

        self.map = map
        self.rows = map.shape[1]
        self.cols = map.shape[0]

        self.original_man_row = man_row
        self.original_man_col = man_col
        self.original_boxes = boxes
        self._initial_state = SokobanState(man_row, man_col, boxes)
        self.initial_state_t = self.state_tensor(self.initial_state)

    @property
    def initial_state(self) -> SokobanState:
        return self._initial_state

    def _parse_string(
        self, string_state: str
    ) -> Optional[tuple[np.ndarray, int, int, np.ndarray]]:
        if len(string_state) > 0:
            cols = len(string_state[0])
            rows = len(string_state)

            map = np.zeros((rows, cols, 2), dtype=np.float32)
            boxes = np.zeros((rows, cols), dtype=np.float32)
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
    def state_width(cls) -> int:
        return 10

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

        if self.map[man_row, man_col + 1, self._channel_walls] == 0:
            actions.append(FourDir.RIGHT)

        if self.map[man_row, man_col - 1][self._channel_walls] == 0:
            actions.append(FourDir.LEFT)

        if self.map[man_row + 1, man_col, self._channel_walls] == 0:
            actions.append(FourDir.DOWN)

        if self.map[man_row - 1, man_col, self._channel_walls] == 0:
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
        boxes = np.argwhere(self.map[:, :, self._channel_boxes])
        map = np.array(self.map)
        domains = []
        for box in boxes:
            if not map[
                box[0] - 1, box[1], [self._channel_walls, self._channel_boxes]
            ].any():
                new_domain = Sokoban(map, box[0] - 1, box[1], boxes, forward=False)
                domains.append(new_domain)

            if not map[
                box[0] + 1, box[1], [self._channel_walls, self._channel_boxes]
            ].any():
                new_domain = Sokoban(map, box[0] + 1, box[1], boxes, forward=False)
                domains.append(new_domain)

            if not map[
                box[0], box[1] + 1, [self._channel_walls, self._channel_boxes]
            ].any():
                new_domain = Sokoban(map, box[0], box[1] + 1, boxes, forward=False)
                domains.append(new_domain)

            if not map[
                box[0], box[1] - 1, [self._channel_walls, self._channel_boxes]
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
        return self.map[:, :, self._channel_goals][
            state.boxes[:, 0], state.boxes[:, 1]
        ].all()

    def state_tensor(self, state: SokobanState) -> to.Tensor:
        channel_man = np.zeros((self.cols, self.rows))
        channel_man[state.man_row, state.man_col] = 1

        channel_boxes = np.zeros((self.cols, self.rows))
        channel_boxes[state.boxes[:, 0], state.boxes[:, 1]] = 1

        arr = np.concatenate((self.map, channel_boxes, channel_man), axis=-1)
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


def parse_problemset(problemset: dict):
    width = problemset["width"]
    goal_tiles = np.arange(width**2).reshape(width, width)

    def parse_specs(problem_specs):
        problems = []
        for spec in problem_specs:
            init_tiles = np.array(spec["tiles"])
            problem = Problem(
                id=spec["id"],
                domain=SlidingTilePuzzle(init_tiles=init_tiles, goal_tiles=goal_tiles),
            )
            problems.append(problem)
        return problems

    model_args = {
        "num_actions": problemset["num_actions"],
        "in_channels": problemset["in_channels"],
        "state_t_width": problemset["state_t_width"],
        "width": problemset["width"],
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
