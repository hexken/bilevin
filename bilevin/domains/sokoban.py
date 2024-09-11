from __future__ import annotations

import numpy as np
import torch as to

from domains.domain import Domain
from domains.state import State
from enums import ActionDir


def get_goal_state(map: np.ndarray) -> SokobanState:
    boxes = np.array(map[Sokoban.box_goal_channel])
    r, c = np.where(map[Sokoban.man_goal_channel])
    r = r.item()
    c = c.item()
    return SokobanState(r, c, boxes)


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


# todo how to set man_goal_channel for start map!
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
        start_state: SokobanState,
        map: np.ndarray,
        forward: bool = True,
    ) -> None:
        super().__init__(forward=forward)
        self.map = map
        self.start_state = start_state

        self.goal_state: SokobanState
        self.goal_state_t: to.Tensor | None = None

    def init(self) -> State:
        self.width = self.map.shape[1]

        if self.forward:
            self.goal_state = get_goal_state(self.map)
            self.goal_state_t = self.state_tensor(self.goal_state)
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
        state = get_goal_state(np.array(self.map))
        domain = Sokoban(state, self.map, forward=False)
        domain.actions = domain._backward_actions
        domain.actions_unpruned = domain._backward_actions_unpruned
        domain.goal_state = self.start_state
        domain.goal_state_t = self.state_tensor(self.start_state)
        return domain

    def is_goal(self, state: SokobanState) -> bool:
        return (state.boxes == self.map[Sokoban.box_goal_channel]).all() and self.map[
            Sokoban.man_goal_channel, state.man_row, state.man_col
        ] == 1

    def state_tensor(self, state: SokobanState) -> to.Tensor:
        channel_man = np.zeros((self.width, self.width), dtype=np.float64)
        channel_man[state.man_row, state.man_col] = 1

        arr = np.concatenate(
            (self.map, state.boxes[None, ...], channel_man[None, ...]),
            axis=0,
            dtype=np.float64,
        )
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
