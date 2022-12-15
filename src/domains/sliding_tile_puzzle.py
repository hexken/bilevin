import copy
from enum import IntEnum
import math

import numpy as np
import torch as to

from domains.environment import Environment


class SlidingTilePuzzle(Environment):
    # todo vectorize this by using a tensor for maintaining state and permuting
    # todo maybe better to have an execution class that handles the execution of the environment

    @property
    @classmethod
    def num_actions(cls):
        return 4

    class Action(IntEnum):
        RIGHT = 0
        LEFT = 1
        UP = 2
        DOWN = 3

    UP = Action.UP
    DOWN = Action.DOWN
    LEFT = Action.LEFT
    RIGHT = Action.RIGHT

    reverse_action = {
        Action.UP: Action.DOWN,
        Action.DOWN: Action.UP,
        Action.LEFT: Action.RIGHT,
        Action.RIGHT: Action.LEFT,
    }

    def __init__(self, tiles):

        if isinstance(tiles, str):
            tiles = tiles.replace("\n", "").split(" ")
            self._size = len(tiles)
            self._tiles = np.zeros(self._size, dtype=np.int32)

            for i, tile in enumerate(tiles):
                if tile == "":
                    continue
                if tile == "B":
                    self._tiles[i] = 0
                else:
                    self._tiles[i] = int(tile)
        elif isinstance(tiles, np.ndarray):
            self._tiles = tiles.copy()
            self._size = len(tiles)
        else:
            raise NotImplementedError("tiles must be specified by a string")

        self._width = int(math.sqrt(self._size))
        self._pos = np.zeros(self._size, dtype=np.int32)

        for i in range(self._size):
            self._pos[self._tiles[i]] = i

            if self._tiles[i] == 0:
                self._blank = i

        self._goal = np.arange(self._size, dtype=np.int32)

    @property
    def in_channels(self):
        return self._size

    def get_backward_problem(self):
        problem = SlidingTilePuzzle(self._tiles)
        problem._goal, problem._tiles = self._tiles, self._goal
        problem._size = self._size
        problem._width = self._width
        problem._pos = np.zeros(self._size, dtype=np.int32)

        for i in range(self._size):
            self._pos[self._tiles[i]] = i

            if self._tiles[i] == 0:
                self._blank = i
        return problem

    def is_valid(self):
        t = 0

        if not (self._width & 1) > 0:
            t = self._pos[0] // self._width

        for i in range(2, self._size):
            for l in range(1, i):
                if self._pos[i] < self._pos[l]:
                    t += 1

        return (int(t) & 1) ^ 1 == 1

    def copy(self):
        return copy.deepcopy(self)

    def getWidth(self):
        return self._width

    def getValueTile(self, i):
        return self._tiles[i]

    def __hash__(self):
        return hash("".join((str(t) for t in self._tiles)))

    def state_equal(self, other):
        return np.array_equal(self._tiles, other._tiles)

    def __eq__(self, other):
        return np.array_equal(self._tiles, other._tiles)

    def successors(self):
        actions = []

        if not ((self._blank + 1) % self._width == 0):  # and op != self._W:
            actions.append(self.RIGHT)

        if self._blank > self._width - 1:  # and op != self._S:
            actions.append(self.UP)

        if not ((self._blank) % self._width == 0):  # and op != self._E:
            actions.append(self.LEFT)

        if self._blank < self._size - self._width:  # and op != self._N:
            actions.append(self.DOWN)

        return actions

    def successors_parent_pruning(self, op):
        actions = []

        if not ((self._blank + 1) % self._width == 0) and op != self.LEFT:
            actions.append(self.RIGHT)

        if self._blank > self._width - 1 and op != self.DOWN:
            actions.append(self.UP)

        if not ((self._blank) % self._width == 0) and op != self.RIGHT:
            actions.append(self.LEFT)

        if self._blank < self._size - self._width and op != self.UP:
            actions.append(self.DOWN)

        return actions

    def apply_action(self, action):

        if action == self.UP:
            self._tiles[self._blank] = self._tiles[self._blank - self._width]
            self._pos[self._tiles[self._blank - self._width]] = self._blank
            self._tiles[self._blank - self._width] = 0
            self._pos[0] = self._blank - self._width
            self._blank = self._blank - self._width

        elif action == self.DOWN:
            self._tiles[self._blank] = self._tiles[self._blank + self._width]
            self._pos[self._tiles[self._blank + self._width]] = self._blank
            self._tiles[self._blank + self._width] = 0
            self._pos[0] = self._blank + self._width
            self._blank = self._blank + self._width

        elif action == self.RIGHT:
            self._tiles[self._blank] = self._tiles[self._blank + 1]
            self._pos[self._tiles[self._blank + 1]] = self._blank
            self._tiles[self._blank + 1] = 0
            self._pos[0] = self._blank + 1
            self._blank = self._blank + 1

        elif action == self.LEFT:
            self._tiles[self._blank] = self._tiles[self._blank - 1]
            self._pos[self._tiles[self._blank - 1]] = self._blank
            self._tiles[self._blank - 1] = 0
            self._pos[0] = self._blank - 1
            self._blank = self._blank - 1
        else:
            raise ValueError(f"Invalid action: {action}")

    def is_solution(self):
        return np.array_equal(self._tiles, self._goal)

    def get_image_representation(self):

        image = np.zeros((self._size, self._width, self._width), dtype=np.float32)

        for num in range(self._size):
            row = int(self._pos[num] / self._width)
            col = int(self._pos[num] % self._width)

            image[num, row, col] = 1

        return image

    def state_tensor(self, device=to.device("cpu")):

        image = to.zeros((self._size, self._width, self._width))
        arr = np.asarray(image)

        for num in range(self._size):
            row = int(self._pos[num] / self._width)
            col = int(self._pos[num] % self._width)

            arr[num, row, col] = 1

        image = image.to(device)
        return image

    def heuristic_value(self):
        h = 0

        for i in range(0, self._size):
            if self._tiles[i] == 0:
                continue
            h = (
                h
                + abs((self._tiles[i] % self._width) - (i % self._width))
                + abs(int((self._tiles[i] / self._width)) - int((i / self._width)))
            )

        return h

    def print(self):
        for i in range(len(self._tiles)):
            print(self._tiles[i], end=" ")
            if (i + 1) % self._width == 0:
                print()

    def one_line(self):
        return " ".join(str(t) for t in self._tiles)
