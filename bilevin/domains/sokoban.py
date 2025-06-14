from __future__ import annotations

import numpy as np
import torch as to

from domains.domain import Domain
from domains.state import State
from enums import ActionDir


class SokobanState(State):
    def __init__(self, man_row: int, man_col: int, boxes: np.ndarray):
        self.man_row = man_row
        self.man_col = man_col
        self.boxes = boxes
        self._hash = (self.man_row, self.man_col, self.boxes.tobytes()).__hash__()

    def __eq__(self, other) -> bool:
        return (
            self.man_row == other.man_row
            and self.man_col == other.man_col
            and (self.boxes == other.boxes).all()
        )

    def __hash__(self) -> int:
        return self._hash


class Sokoban(Domain):
    goal_str = "."
    man_str = "@"
    wall_str = "#"
    box_str = "$"
    man_goal_str = "G"

    wall_channel = 0
    box_goal_channel = 1
    man_goal_channel = 2

    def __init__(
        self,
        start_state: SokobanState | list[SokobanState],
        map: np.ndarray,
        forward: bool = True,
    ) -> None:
        super().__init__(forward=forward)
        self.map = map
        self.start_state = start_state

        self.goal_state: SokobanState | None = None
        self.goal_state_t: to.Tensor | None = None

    def init(self) -> SokobanState | list[SokobanState]:
        self.width = self.map.shape[1]
        self.man_goal_locs = set()
        for r, c in zip(*np.where(self.map[Sokoban.man_goal_channel])):
            r = r.item()
            c = c.item()
            self.man_goal_locs.add((r, c))
        return self._init()

    @property
    def state_t_width(self) -> int:
        return 10

    @property
    def state_t_depth(self) -> int:
        return 1

    @property
    def num_actions(self) -> int:
        return 4

    @property
    def in_channels(self) -> int:
        return 5

    def actions_unpruned(self, state: SokobanState) -> list[ActionDir]:
        """
        Can move in each direction where either there is no wall or box, or there is a box that can
        be pushed (i.e. no wall or box behind it)
        """
        actions = []
        man_row = state.man_row
        man_col = state.man_col

        if (
            self.map[Sokoban.wall_channel, man_row, man_col + 1] == 0
            and state.boxes[man_row, man_col + 1] == 0
        ):
            actions.append(ActionDir.RIGHT)
        elif (
            state.boxes[man_row, man_col + 1] == 1
            and self.map[Sokoban.wall_channel, man_row, man_col + 2] == 0
            and state.boxes[man_row, man_col + 2] == 0
        ):
            actions.append(ActionDir.RIGHT)

        if (
            self.map[Sokoban.wall_channel, man_row, man_col - 1] == 0
            and state.boxes[man_row, man_col - 1] == 0
        ):
            actions.append(ActionDir.LEFT)
        elif (
            state.boxes[man_row, man_col - 1] == 1
            and self.map[Sokoban.wall_channel, man_row, man_col - 2] == 0
            and state.boxes[man_row, man_col - 2] == 0
        ):
            actions.append(ActionDir.LEFT)

        if (
            self.map[Sokoban.wall_channel, man_row + 1, man_col] == 0
            and state.boxes[man_row + 1, man_col] == 0
        ):
            actions.append(ActionDir.DOWN)
        elif (
            state.boxes[man_row + 1, man_col] == 1
            and self.map[Sokoban.wall_channel, man_row + 2, man_col] == 0
            and state.boxes[man_row + 2, man_col] == 0
        ):
            actions.append(ActionDir.DOWN)

        if (
            self.map[Sokoban.wall_channel, man_row - 1, man_col] == 0
            and state.boxes[man_row - 1, man_col] == 0
        ):
            actions.append(ActionDir.UP)
        elif (
            state.boxes[man_row - 1, man_col] == 1
            and self.map[Sokoban.wall_channel, man_row - 2, man_col] == 0
            and state.boxes[man_row - 2, man_col] == 0
        ):
            actions.append(ActionDir.UP)

        return actions

    def actions(self, parent_action: ActionDir, state: SokobanState) -> list[ActionDir]:
        return self.actions_unpruned(state)

    def result(self, state: SokobanState, action: ActionDir) -> SokobanState:
        boxes = np.array(state.boxes)
        man_row = state.man_row
        man_col = state.man_col

        if action == ActionDir.UP:
            if boxes[man_row - 1, man_col] == 1:
                boxes[man_row - 1, man_col] = 0
                boxes[man_row - 2, man_col] = 1
            man_row -= 1

        elif action == ActionDir.DOWN:
            if boxes[man_row + 1, man_col] == 1:
                boxes[man_row + 1, man_col] = 0
                boxes[man_row + 2, man_col] = 1
            man_row += 1

        elif action == ActionDir.RIGHT:
            if boxes[man_row, man_col + 1] == 1:
                boxes[man_row, man_col + 1] = 0
                boxes[man_row, man_col + 2] = 1
            man_col += 1

        elif action == ActionDir.LEFT:
            if boxes[man_row, man_col - 1] == 1:
                boxes[man_row, man_col - 1] = 0
                boxes[man_row, man_col - 2] = 1
            man_col -= 1

        return SokobanState(man_row, man_col, boxes)

    def reverse_action(self, action: ActionDir) -> ActionDir:
        if action == ActionDir.UP:
            return ActionDir.DOWN
        elif action == ActionDir.DOWN:
            return ActionDir.UP
        elif action == ActionDir.LEFT:
            return ActionDir.RIGHT
        elif action == ActionDir.RIGHT:
            return ActionDir.LEFT

    def _backward_actions_unpruned(self, state: SokobanState) -> list[ActionDir]:
        assert not self.forward
        actions = []
        man_row = state.man_row
        man_col = state.man_col

        if (
            self.map[Sokoban.wall_channel, man_row, man_col + 1] == 0
            and state.boxes[man_row, man_col + 1] == 0
        ):
            actions.append(ActionDir.RIGHT)

        if (
            self.map[Sokoban.wall_channel, man_row, man_col - 1] == 0
            and state.boxes[man_row, man_col - 1] == 0
        ):
            actions.append(ActionDir.LEFT)

        if (
            self.map[Sokoban.wall_channel, man_row + 1, man_col] == 0
            and state.boxes[man_row + 1, man_col] == 0
        ):
            actions.append(ActionDir.DOWN)

        if (
            self.map[Sokoban.wall_channel, man_row - 1, man_col] == 0
            and state.boxes[man_row - 1, man_col] == 0
        ):
            actions.append(ActionDir.UP)

        return actions

    def _backward_actions(
        self, parent_action: ActionDir, state: SokobanState
    ) -> list[ActionDir]:
        return self._backward_actions_unpruned(state)

    def _backward_result(self, state: SokobanState, action: ActionDir) -> SokobanState:
        boxes = np.array(state.boxes)
        man_row = state.man_row
        man_col = state.man_col

        if action == ActionDir.UP:
            if boxes[man_row + 1, man_col] == 1:
                boxes[man_row + 1, man_col] = 0
                boxes[man_row, man_col] = 1
            man_row -= 1

        elif action == ActionDir.DOWN:
            if boxes[man_row - 1, man_col] == 1:
                boxes[man_row - 1, man_col] = 0
                boxes[man_row, man_col] = 1
            man_row += 1

        elif action == ActionDir.RIGHT:
            if boxes[man_row, man_col - 1] == 1:
                boxes[man_row, man_col - 1] = 0
                boxes[man_row, man_col] = 1
            man_col += 1

        elif action == ActionDir.LEFT:
            if boxes[man_row, man_col + 1] == 1:
                boxes[man_row, man_col + 1] = 0
                boxes[man_row, man_col] = 1
            man_col -= 1

        return SokobanState(man_row, man_col, boxes)

    def backward_domain(self) -> Sokoban:
        """
        create new states by placing the boxes on the goals, then the man at each side of each box, where not blocked
        """
        assert self.forward
        map = np.zeros((3, self.width, self.width), dtype=np.float64)
        map[0, :] = self.map[0, :]
        map[1, :] = self.start_state.boxes[:]

        man_goal_row, man_goal_col = self.start_state.man_row, self.start_state.man_col
        map[2, man_goal_row, man_goal_col] = 1

        boxes = self.map[Sokoban.box_goal_channel]

        start_states = []
        for r, c in zip(*np.where(self.map[Sokoban.man_goal_channel])):
            r = r.item()
            c = c.item()
            state = SokobanState(r, c, boxes)
            start_states.append(state)

        domain = Sokoban(start_states, map, forward=False)
        domain.actions = domain._backward_actions
        domain.actions_unpruned = domain._backward_actions_unpruned
        domain.result = domain._backward_result
        return domain

    def is_goal(self, state: SokobanState) -> bool:
        return (state.boxes == self.map[Sokoban.box_goal_channel]).all() and (
            (state.man_row, state.man_col) in self.man_goal_locs
        )

    def state_tensor(self, state: SokobanState) -> to.Tensor:
        arr = np.zeros((5, self.width, self.width), dtype=np.float64)
        arr[:3] = self.map[:]
        arr[3, :] = state.boxes[:]
        arr[4, state.man_row, state.man_col] = 1

        return to.from_numpy(arr)

    def print(self, state: SokobanState):
        width = self.map.shape[1]
        for i in range(width):
            for j in range(width):
                if (
                    self.map[
                        Sokoban.box_goal_channel,
                        i,
                        j,
                    ]
                    == 1
                ):
                    if state.boxes[i, j] == 1:
                        print("*", end="")
                    else:
                        print(Sokoban.goal_str, end="")
                elif self.map[Sokoban.man_goal_channel, i, j] == 1:
                    if i == state.man_row and j == state.man_col:
                        print("X", end="")
                    else:
                        print(Sokoban.man_goal_str, end="")
                elif i == state.man_row and j == state.man_col:
                    print(Sokoban.man_str, end="")
                elif (
                    self.map[
                        Sokoban.wall_channel,
                        i,
                        j,
                    ]
                    == 1
                ):
                    print(Sokoban.wall_str, end="")
                elif state.boxes[i, j] == 1:
                    print(Sokoban.box_str, end="")
                else:
                    print(" ", end="")
            print()
