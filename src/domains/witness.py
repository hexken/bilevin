from collections import deque
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch as to

from domains.domain import Domain, State
from enums import Color, FourDir


class WitnessState(State):
    def __init2__(
        self,
        head_dot_row,
        head_dot_col,
        goal_dot_row,
        goal_dot_col,
        state_list,
        num_rows: int = 4,
        num_cols: int = 4,
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cells = np.zeros((num_rows, num_cols))

        self.dots = np.zeros((num_rows + 1, num_cols + 1))
        self.head_x = head_dot_row
        self.head_y = head_dot_col
        self.goal_x = goal_dot_row
        self.goal_y = goal_dot_col

        self.v_segs = np.zeros((num_rows, num_cols + 1))
        self.h_segs = np.zeros((num_rows + 1, num_cols))

        # self.read_state_from_list(state_list)

    def __init__(self, puzzle, max_rows: int = 11, max_cols: int = 11):
        self.max_rows = max_rows
        self.max_cols = max_cols

        values = puzzle[0].replace("Size: ", "").split(" ")
        self.num_rows = int(values[0])
        self.num_cols = int(values[1])

        values = puzzle[1].replace("Init: ", "").split(" ")
        self.head_y = int(values[0])
        self.head_x = int(values[1])
        self.start_y = self.head_y
        self.start_x = self.head_x

        values = puzzle[2].replace("Goal: ", "").split(" ")
        self.goal_y = int(values[0])
        self.goal_x = int(values[1])

        self.v_segs = np.zeros((self.num_rows, self.num_cols + 1))
        self.h_segs = np.zeros((self.num_rows + 1, self.num_cols))
        self.dots = np.zeros((self.num_rows + 1, self.num_cols + 1))
        self.dots[self.head_y][self.head_x] = 1

        self.cells = np.zeros((self.num_rows, self.num_cols))
        values = puzzle[3].replace("Colors: |", "").split("|")
        for t in values:
            numbers = t.split(" ")
            self.cells[int(numbers[0])][int(numbers[1])] = int(numbers[2])

    def ishead_at_goal(self):
        return self.head_y == self.goal_y and self.head_x == self.goal_x

    def as_tensor(self, device=to.device("cpu")):
        """
        Generates an image representation for the puzzle. Currently the method supports 4 colors and includes
        the following channels (third dimension of image): one channel for each color; one channel with 1's
        where is "open" in the grid (this allows learning systems to work with a fixed image size defined
        by max_lines and max_columns); one channel for the current path (cells occupied by the snake);
        one channel for the tip of the snake; one channel for the exit of the puzzle; one channel for the
        entrance of the snake. In total there are 9 different channels.

        Each channel is a matrix with zeros and ones. The image returned is a 3-dimensional numpy array.
        """

        number_of_colors = 4
        channels = 9

        # defining the 3-dimnesional array that will be filled with the puzzle's information
        image = to.zeros(channels, 2 * self.max_rows, 2 * self.max_cols)
        arr = np.asarray(image)

        # create one channel for each color i
        for i in range(0, number_of_colors):
            for j in range(0, self.cells.shape[0]):
                for k in range(0, self.cells.shape[1]):
                    if self.cells[j][k] == i:
                        arr[i][2 * j + 1][2 * k + 1] = 1
        channel_number = number_of_colors

        # the number_of_colors-th channel specifies the open spaces in the grid
        for j in range(0, 2 * self.num_rows + 1):
            for k in range(0, 2 * self.num_cols + 1):
                arr[channel_number][j][k] = 1

        # channel for the current path
        channel_number += 1
        for i in range(0, self.v_segs.shape[0]):
            for j in range(0, self.v_segs.shape[1]):
                if self.v_segs[i][j] == 1:
                    arr[channel_number][2 * i][2 * j] = 1
                    arr[channel_number][2 * i + 1][2 * j] = 1
                    arr[channel_number][2 * i + 2][2 * j] = 1

        for i in range(0, self.h_segs.shape[0]):
            for j in range(0, self.h_segs.shape[1]):
                if self.h_segs[i][j] == 1:
                    arr[channel_number][2 * i][2 * j] = 1
                    arr[channel_number][2 * i][2 * j + 1] = 1
                    arr[channel_number][2 * i][2 * j + 2] = 1

        # channel with the tip of the snake
        channel_number += 1
        arr[channel_number][2 * self.head_y][2 * self.head_x] = 1

        # channel for the exit of the puzzle
        channel_number += 1
        arr[channel_number][2 * self.goal_y][2 * self.goal_x] = 1

        # channel for the entrance of the puzzle
        channel_number += 1
        arr[channel_number][2 * self.head_y][2 * self.head_x] = 1

        image = image.to(device)
        return image

    def __repr__(self):
        state_str = "Cells: \n"
        state_str += "\n".join(
            "\t".join("%d" % x for x in y) for y in reversed(self.cells)
        )
        state_str += "\nDots: \n"
        state_str += "\n".join(
            "\t".join("%d" % x for x in y) for y in reversed(self.dots)
        )
        return state_str

    def __hash__(self):
        return hash((str(self.v_segs), str(self.h_segs), str(self.cells)))

    def __eq__(self, other):
        return (
            np.array_equal(self.v_segs, other.v_segs)
            and np.array_equal(self.h_segs, other.h_segs)
            and np.array_equal(self.cells, other.cells)
            and np.array_equal(self.dots, other.dots)
        )

    def print(self):
        image = self.as_tensor()
        for i in range(0, image.shape[2]):
            for j in range(0, image.shape[0]):
                for k in range(0, image.shape[0]):
                    print(image[j][k][i], end=" ")
                print()
            print("\n\n")

    def plot(self, filename=None):
        """
        This method plots the state. Several features in this method are hard-coded and might
        need adjustment as one changes the size of the puzzle. For example, the size of the figure is set to be fixed
        to [5, 5] (see below).
        """
        fig, (ax,) = plt.subplots(figsize=(5, 5))
        # fig.patch.set_facecolor((1, 1, 1))

        # draw vertical lines of the grid
        for y in range(self.dots.shape[1]):
            ax.plot([y, y], [0, self.cells.shape[0]], Color.BLACK)
        # draw horizontal lines of the grid
        for x in range(self.dots.shape[0]):
            ax.plot([0, self.cells.shape[1]], [x, x], Color.BLACK)

        # scale the axis area to fill the whole figure
        ax.set_position([0, 0, 1, 1])

        ax.set_axis_off()

        ax.set_xlim(-1, np.max(self.dots.shape))
        ax.set_ylim(-1, np.max(self.dots.shape))

        # Draw the vertical segments of the path
        for i in range(self.v_segs.shape[0]):
            for j in range(self.v_segs.shape[1]):
                if self.v_segs[i][j] == 1:
                    ax.plot([j, j], [i, i + 1], Color.RED, linewidth=5)

        # Draw the horizontal segments of the path
        for i in range(self.h_segs.shape[0]):
            for j in range(self.h_segs.shape[1]):
                if self.h_segs[i][j] == 1:
                    ax.plot([j, j + 1], [i, i], Color.RED, linewidth=5)

        # Draw the separable bullets according to the values in self.cells and Color enum type
        offset = 0.5
        for i in range(self.cells.shape[0]):
            for j in range(self.cells.shape[1]):
                if self.cells[i][j] != 0:
                    ax.plot(
                        j + offset,
                        i + offset,
                        "o",
                        markersize=15,
                        markeredgecolor=(0, 0, 0),
                        markerfacecolor=Color.int_values()[int(self.cells[i][j] - 1)],
                        markeredgewidth=2,
                    )

        # Draw the intersection of lines: red for an intersection that belongs to a path and black otherwise
        for i in range(self.dots.shape[0]):
            for j in range(self.dots.shape[1]):
                if self.dots[i][j] != 0:
                    ax.plot(
                        j,
                        i,
                        "o",
                        markersize=10,
                        markeredgecolor=(0, 0, 0),
                        markerfacecolor=Color.RED,
                        markeredgewidth=0,
                    )
                else:
                    ax.plot(
                        j,
                        i,
                        "o",
                        markersize=10,
                        markeredgecolor=(0, 0, 0),
                        markerfacecolor=Color.BLACK,
                        markeredgewidth=0,
                    )

        # Draw the entrance of the puzzle in red as it is always on the state's path
        ax.plot(
            self.start_x - 0.15,
            self.start_y,
            ">",
            markersize=10,
            markeredgecolor=(0, 0, 0),
            markerfacecolor=Color.RED,
            markeredgewidth=0,
        )

        column_exit_offset = 0
        row_exit_offset = 0

        if self.goal_x == self.num_cols:
            column_exit_offset = 0.15
            exit_symbol = ">"
        elif self.goal_x == 0:
            column_exit_offset = -0.15
            exit_symbol = "<"
        elif self.goal_y == self.num_rows:
            row_exit_offset = 0.15
            exit_symbol = "^"
        else:
            row_exit_offset = -0.15
            exit_symbol = "v"
        # Draw the exit of the puzzle: red if it is on a path, black otherwise
        if self.dots[self.goal_y][self.goal_x] == 0:
            ax.plot(
                self.goal_x + column_exit_offset,
                self.goal_y + row_exit_offset,
                exit_symbol,
                markersize=10,
                markeredgecolor=(0, 0, 0),
                markerfacecolor=Color.BLACK,
                markeredgewidth=0,
            )
        else:
            ax.plot(
                self.goal_x + column_exit_offset,
                self.goal_y + row_exit_offset,
                exit_symbol,
                markersize=10,
                markeredgecolor=(0, 0, 0),
                markerfacecolor=Color.RED,
                markeredgewidth=0,
            )

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()


class Witness(Domain):
    def __init__(
        self,
        initial_state_list,
        max_rows=11,
        max_cols=11,
    ):
        self.initial_state = WitnessState(initial_state_list, max_rows, max_cols)

        self.start_y = self.initial_state.start_y
        self.start_x = self.initial_state.start_x
        self.goal_y = self.initial_state.goal_y
        self.goal_x = self.initial_state.goal_x
        self.num_rows = self.initial_state.num_rows
        self.num_cols = self.initial_state.num_cols
        self.max_rows = self.initial_state.max_rows
        self.max_cols = self.initial_state.max_cols
        assert self.max_rows == self.max_cols

        self.cells = None
        self.all_dots = None
        self.all_v_segs = None
        self.all_h_segs = None

    def __init2__(
        self,
        start_y,
        start_x,
        goal_y,
        goal_x,
        initial_state_list,
        num_rows=4,
        num_cols=4,
        max_rows=11,
        max_cols=11,
    ):
        self.start_y = start_y
        self.start_x = start_x
        self.goal_y = goal_y
        self.goal_x = goal_x
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.max_rows = max_rows
        self.max_cols = max_cols
        assert self.max_rows == self.max_cols

        self.cells = None
        self.all_dots = None
        self.all_v_segs = None
        self.all_h_segs = None

        self.initial_state = WitnessState(initial_state_list, max_rows, max_cols)

    def reset(self):
        self.cells = self.initial_state.cells.copy()
        self.all_dots = self.initial_state.dots.copy()
        self.all_v_segs = self.initial_state.v_segs.copy()
        self.all_h_segs = self.initial_state.h_segs.copy()

        return self.initial_state  # todo might need deeopcopy, check mutability reqs

    @property
    def num_actions(cls):
        return 4

    @property
    def in_channels(self):
        return 9

    @property
    def state_size(self):
        """side length"""
        return self.max_rows * 2

    def reverse_action(self, action: FourDir):
        if action == FourDir.UP:
            return FourDir.DOWN
        elif action == FourDir.DOWN:
            return FourDir.UP
        elif action == FourDir.LEFT:
            return FourDir.RIGHT
        elif action == FourDir.RIGHT:
            return FourDir.LEFT

    def successors_parent_pruning(self, parent_action, state):
        """
        Successor function used by planners trying to solve the puzzle. The method returns
        a list with legal actions for the state. The valid actions for the domain are {U, D, L, R}.

        The tip of the snake can move to an adjacent intersection in the grid as long as
        that intersection isn't already occupied by the snake and the intersection is valid
        (i.e., it isn't negative or larger than the grid size)

        op is the action taken at the parent; used here to perform parent pruning

        Mapping of actions:
        0 - Up
        1 - Down
        2 - Right
        3 - Left
        """
        actions = []
        #         if self.has_tip_reached_goal():
        #             return actions
        # moving up
        if (
            parent_action != FourDir.DOWN
            and state.head_y + 1 < state.dots.shape[0]
            and state.v_segs[state.head_y][state.head_x] == 0
            and state.dots[state.head_y + 1][state.head_x] == 0
        ):
            actions.append(FourDir.UP)
        # moving down
        if (
            parent_action != FourDir.UP
            and state.head_y - 1 >= 0
            and state._v_seg[state.head_y - 1][state.head_x] == 0
            and state._dots[state.head_y - 1][state.head_x] == 0
        ):
            actions.append(FourDir.DOWN)
        # moving right
        if (
            parent_action != FourDir.LEFT
            and state.head_x + 1 < state._dots.shape[1]
            and state._h_seg[state.head_y][state.head_x] == 0
            and state._dots[state.head_y][state.head_x + 1] == 0
        ):
            actions.append(FourDir.RIGHT)
        # moving left
        if (
            parent_action != FourDir.RIGHT
            and state.head_x - 1 >= 0
            and state._h_seg[state.head_y][state.head_x - 1] == 0
            and state._dots[state.head_y][state.head_x - 1] == 0
        ):
            actions.append(FourDir.LEFT)

        return actions

    def successors(self, state):
        """
        Successor function used by planners trying to solve the puzzle. The method returns
        a list with legal actions for the state. The valid actions for the domain are {U, D, L, R}.

        The tip of the snake can move to an adjacent intersection in the grid as long as
        that intersection isn't already occupied by the snake and the intersection is valid
        (i.e., it isn't negative or larger than the grid size)

        Mapping of actions:
        0 - Up
        1 - Down
        2 - Right
        3 - Left
        """
        actions = []
        #         if self.has_tip_reached_goal():
        #             return actions
        # moving up
        if (
            state.head_y + 1 < state._dots.shape[0]
            and state.v_segs[state.head_y][state.head_x] == 0
            and state.dots[state.head_y + 1][state.head_x] == 0
        ):
            actions.append(FourDir.UP)
        # moving down
        if (
            state.head_y - 1 >= 0
            and state.v_segs[state.head_y - 1][state.head_x] == 0
            and state.dots[state.head_y - 1][state.head_x] == 0
        ):
            actions.append(FourDir.DOWN)
        # moving right
        if (
            state.head_x + 1 < state._dots.shape[1]
            and state.h_segs[state.head_y][state.head_x] == 0
            and state.dots[state.head_y][state.head_x + 1] == 0
        ):
            actions.append(FourDir.RIGHT)
        # moving left
        if (
            state.head_x - 1 >= 0
            and state.h_segs[state.head_y][state.head_x - 1] == 0
            and state.dots[state.head_y][state.head_x - 1] == 0
        ):
            actions.append(FourDir.LEFT)

        return actions

    def apply_action(self, state, action: FourDir):
        """
        Applies a given action to the state. It modifies the segments visited by the snake (v_seg and h_seg),
        the intersections (dots), and the tip of the snake.
        """
        new_state = deepcopy(state)

        # moving up
        if action == FourDir.UP:
            new_state.v_segs[new_state.head_y][new_state.head_x] = 1
            new_state.dots[new_state.head_y + 1][new_state.head_x] = 1
            new_state.head_y += 1
        # moving down
        if action == FourDir.DOWN:
            new_state.v_segs[new_state.head_y - 1][new_state.head_x] = 1
            new_state.dots[new_state.head_y - 1][new_state.head_x] = 1
            new_state.head_y -= 1
        # moving right
        if action == FourDir.RIGHT:
            new_state.h_segs[new_state.head_y][new_state.head_x] = 1
            new_state.dots[new_state.head_y][new_state.head_x + 1] = 1
            new_state.head_x += 1
        # moving left
        if action == FourDir.LEFT:
            new_state.h_segs[new_state.head_y][new_state.head_x - 1] = 1
            new_state.dots[new_state.head_y][new_state.head_x - 1] = 1
            new_state.head_x -= 1

        return new_state

    def is_solution(self, state):
        """
        Verifies whether the state's path represents a valid solution. This is performed by verifying the following
        (1) the tip of the snake is at the goal position
        (2) a bullet of color c1 cannot reach a bullet of color c2 through a BFS search.

        The BFS uses the cells (line and column) as states and verifies whether cells with a bullet of a given color
        can only reach (and be reached) by cells with bullets of the same color (or of the neutral color, denoted as zero in this implementation)
        """
        if not state.ishead_at_goal():
            return False

        non_visited_states = set()
        current_color = Color.NEUTRAL
        closed_bfs = np.zeros((self.num_rows, self.num_cols))
        for i in range(state.cells.shape[0]):
            for j in range(state.cells.shape[1]):
                non_visited_states.add((i, j))

        while len(non_visited_states) != 0:
            root = non_visited_states.pop()
            # If root of new BFS search was already visited, then go to the next state
            if closed_bfs[root] == 1:
                continue
            current_color = state.cells[root]

            open_bfs = deque()
            open_bfs.append(root)
            closed_bfs[root] = 1
            while len(open_bfs) != 0:
                # remove first cell (state) from queue
                cell = open_bfs.popleft()

                def _successors(self, cell):
                    """
                    Successor function use in the Breadth-first search (BFS) performed to validate a solution.
                    An adjacent cell c' is amongst the successors of cell c if there is no segment (v_seg or h_seg)
                    separating cells c and c'.

                    This method is meant to be called only from within GameState
                    """
                    children = []
                    row, col = cell
                    # move up
                    if (
                        row + 1 < self.cells.shape[0]
                        and state.h_segs[row + 1][col] == 0
                    ):
                        children.append((row + 1, col))
                    # move down
                    if row > 0 and state.h_segs[row][col] == 0:
                        children.append((row - 1, col))
                    # move right
                    if (
                        col + 1 < self.cells.shape[1]
                        and state.v_segs[row][col + 1] == 0
                    ):
                        children.append((row, col + 1))
                    # move left
                    if col > 0 and state.v_segs[row][col] == 0:
                        children.append((row, col - 1))
                    return children

                children = _successors(self, cell)
                for c in children:
                    # If c is a duplicate, then continue with the next child
                    if closed_bfs[c] == 1:
                        continue
                    # If c's color isn't neutral (zero) and it is different from current_color, then state isn't a soution
                    if (
                        current_color != Color.NEUTRAL
                        and state.cells[c] != Color.NEUTRAL
                        and state.cells[c] != current_color
                    ):
                        return False
                    # If current_color is neutral (zero) and c's color isn't, then attribute c's color to current_color
                    if state.cells[c] != Color.NEUTRAL:
                        current_color = state.cells[c]
                    # Add c to BFS's open list
                    open_bfs.append(c)
                    # mark state c as visited
                    closed_bfs[c] = 1
        return True
