from collections import deque
import random
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import torch as to

from domains.domain import DirAction, Domain, State


class Color(Enum):
    BLUE = "b"
    RED = "r"
    GREEN = "g"
    CYAN = "c"
    YELLOW = "y"
    MAGENTA = "m"
    BLACK = "k"


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
        self.head_h_dot = head_dot_row
        self.head_v_dot = head_dot_col
        self.goal_h_dot = goal_dot_row
        self.goal_v_dot = goal_dot_col

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
        self.head_h_dot = int(values[0])
        self.head_v_dot = int(values[1])

        values = puzzle[2].replace("Goal: ", "").split(" ")
        self.goal_h_dot = int(values[0])
        self.goal_v_dot = int(values[1])

        self.cells = np.zeros((self.num_rows, self.num_cols))
        self.v_segs = np.zeros((self.num_rows, self.num_cols + 1))
        self.h_segs = np.zeros((self.num_rows + 1, self.num_cols))
        self.dots = np.zeros((self.num_rows + 1, self.num_cols + 1))
        self.dots[self.head_h_dot][self.head_v_dot] = 1

        values = puzzle[3].replace("Colors: |", "").split("|")
        for t in values:
            numbers = t.split(" ")
            self.cells[int(numbers[0])][int(numbers[1])] = int(numbers[2])

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
        arr[channel_number][2 * self.head_h_dot][2 * self.head_v_dot] = 1

        # channel for the exit of the puzzle
        channel_number += 1
        arr[channel_number][2 * self.goal_h_dot][2 * self.goal_v_dot] = 1

        # channel for the entrance of the puzzle
        channel_number += 1
        arr[channel_number][2 * self.head_h_dot][2 * self.head_v_dot] = 1

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
        fig = plt.figure(figsize=[5, 5])
        fig.patch.set_facecolor((1, 1, 1))

        ax = fig.add_subplot(111)

        # draw vertical lines of the grid
        for y in range(self.dots.shape[1]):
            ax.plot([y, y], [0, self.cells.shape[0]], "k")
        # draw horizontal lines of the grid
        for x in range(self.dots.shape[0]):
            ax.plot([0, self.cells.shape[1]], [x, x], "k")

        # scale the axis area to fill the whole figure
        ax.set_position([0, 0, 1, 1])

        ax.set_axis_off()

        ax.set_xlim(-1, np.max(self.dots.shape))
        ax.set_ylim(-1, np.max(self.dots.shape))

        # Draw the vertical segments of the path
        for i in range(self.v_segs.shape[0]):
            for j in range(self.v_segs.shape[1]):
                if self.v_segs[i][j] == 1:
                    ax.plot([j, j], [i, i + 1], "r", linewidth=5)

        # Draw the horizontal segments of the path
        for i in range(self.h_segs.shape[0]):
            for j in range(self.h_segs.shape[1]):
                if self.h_segs[i][j] == 1:
                    ax.plot([j, j + 1], [i, i], "r", linewidth=5)

        # Draw the separable bullets according to the values in self._cells and self._colors
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
                        markerfacecolor=list(Color)[int(self.cells[i][j] - 1)],
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
            self._column_init - 0.15,
            self._line_init,
            ">",
            markersize=10,
            markeredgecolor=(0, 0, 0),
            markerfacecolor=Color.RED,
            markeredgewidth=0,
        )

        column_exit_offset = 0
        line_exit_offset = 0

        if self._column_goal == self._columns:
            column_exit_offset = 0.15
            exit_symbol = ">"
        elif self._column_goal == 0:
            column_exit_offset = -0.15
            exit_symbol = "<"
        elif self._line_goal == self._lines:
            line_exit_offset = 0.15
            exit_symbol = "^"
        else:
            line_exit_offset = -0.15
            exit_symbol = "v"
        # Draw the exit of the puzzle: red if it is on a path, black otherwise
        if self._dots[self._line_goal][self._column_goal] == 0:
            ax.plot(
                self._column_goal + column_exit_offset,
                self._line_goal + line_exit_offset,
                exit_symbol,
                markersize=10,
                markeredgecolor=(0, 0, 0),
                markerfacecolor="k",
                markeredgewidth=0,
            )
        else:
            ax.plot(
                self._column_goal + column_exit_offset,
                self._line_goal + line_exit_offset,
                exit_symbol,
                markersize=10,
                markeredgecolor=(0, 0, 0),
                markerfacecolor="r",
                markeredgewidth=0,
            )

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def save_state(self, filename):
        """
        Saves the state into filename. It doesn't save the path in the state.

        Here's an example of a file of a puzzle with 5 lines and 4 columns, with the snake
        starting position at 0, 0 and finishing position at 5, 4. The state has three bullets,
        each with a different color (1, 2, and 6). The bullets are located at (0,0), (2,2), (3, 3),
        and (2, 0). Here the first number means the line and the second the column.

        Size: 5 4
        Init: 0 0
        Goal: 5 4
        Colors: |0 0 1|2 2 2|3 3 6|2 0 1

        """
        file = open(filename, "w")
        file.write("Size: " + str(self.num_rows) + " " + str(self.num_cols) + "\n")
        file.write(
            "Init: " + str(self.head_v_dot) + " " + str(self._column_init) + "\n"
        )
        file.write(
            "Goal: " + str(self._line_goal) + " " + str(self._column_goal) + "\n"
        )

        has_colors = False
        for i in range(self._cells.shape[0]):
            for j in range(self._cells.shape[1]):
                if self._cells[i][j] != 0:
                    has_colors = True
                    break
            if has_colors:
                break

        if has_colors:
            file.write("Colors: ")
            for i in range(self._cells.shape[0]):
                for j in range(self._cells.shape[1]):
                    if self._cells[i][j] != 0:
                        file.write(
                            "|"
                            + str(i)
                            + " "
                            + str(j)
                            + " "
                            + str(int(self._cells[i][j]))
                        )
            file.close()


class Witness(Domain):
    _colors = ["b", "r", "g", "c", "y", "m"]

    def __init__(
        self,
        head_dot_row,
        head_dot_col,
        goal_dot_row,
        goal_dot_col,
        initial_state_list,
        num_rows=4,
        num_cols=4,
        max_rows=11,
        max_cols=11,
    ):
        self.head_dot_row = head_dot_row
        self.head_dot_col = head_dot_col
        self.goal_dot_row = goal_dot_row
        self.goal_dot_col = goal_dot_col
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

    def reverse_action(self, action: DirAction):
        if action == DirAction.UP:
            return DirAction.DOWN
        elif action == DirAction.DOWN:
            return DirAction.UP
        elif action == DirAction.LEFT:
            return DirAction.RIGHT
        elif action == DirAction.RIGHT:
            return DirAction.LEFT

    def __successor_bfs(self, state):
        """
        Successor function use in the Breadth-first search (BFS) performed to validate a solution.
        An adjacent cell c' is amongst the successors of cell c if there is no segment (v_seg or h_seg)
        separating cells c and c'.

        This method is meant to be called only from within GameState
        """
        children = []
        # move up
        if (
            state[0] + 1 < self._cells.shape[0]
            and self._h_seg[state[0] + 1][state[1]] == 0
        ):
            children.append((state[0] + 1, state[1]))
        # move down
        if state[0] > 0 and self._h_seg[state[0]][state[1]] == 0:
            children.append((state[0] - 1, state[1]))
        # move right
        if (
            state[1] + 1 < self._cells.shape[1]
            and self._v_seg[state[0]][state[1] + 1] == 0
        ):
            children.append((state[0], state[1] + 1))
        # move left
        if state[1] > 0 and self._v_seg[state[0]][state[1]] == 0:
            children.append((state[0], state[1] - 1))
        return children

    def successors_parent_pruning(self, op):
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
            op != DirAction.DOWN
            and self._line_tip + 1 < self._dots.shape[0]
            and self._v_seg[self._line_tip][self._column_tip] == 0
            and self._dots[self._line_tip + 1][self._column_tip] == 0
        ):
            actions.append(DirAction.UP)
        # moving down
        if (
            op != DirAction.UP
            and self._line_tip - 1 >= 0
            and self._v_seg[self._line_tip - 1][self._column_tip] == 0
            and self._dots[self._line_tip - 1][self._column_tip] == 0
        ):
            actions.append(DirAction.DOWN)
        # moving right
        if (
            op != DirAction.LEFT
            and self._column_tip + 1 < self._dots.shape[1]
            and self._h_seg[self._line_tip][self._column_tip] == 0
            and self._dots[self._line_tip][self._column_tip + 1] == 0
        ):
            actions.append(DirAction.RIGHT)
        # moving left
        if (
            op != DirAction.RIGHT
            and self._column_tip - 1 >= 0
            and self._h_seg[self._line_tip][self._column_tip - 1] == 0
            and self._dots[self._line_tip][self._column_tip - 1] == 0
        ):
            actions.append(DirAction.LEFT)
        return actions

    def successors(self):
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
            self._line_tip + 1 < self._dots.shape[0]
            and self._v_seg[self._line_tip][self._column_tip] == 0
            and self._dots[self._line_tip + 1][self._column_tip] == 0
        ):
            actions.append(DirAction.UP)
        # moving down
        if (
            self._line_tip - 1 >= 0
            and self._v_seg[self._line_tip - 1][self._column_tip] == 0
            and self._dots[self._line_tip - 1][self._column_tip] == 0
        ):
            actions.append(DirAction.DOWN)
        # moving right
        if (
            self._column_tip + 1 < self._dots.shape[1]
            and self._h_seg[self._line_tip][self._column_tip] == 0
            and self._dots[self._line_tip][self._column_tip + 1] == 0
        ):
            actions.append(DirAction.RIGHT)
        # moving left
        if (
            self._column_tip - 1 >= 0
            and self._h_seg[self._line_tip][self._column_tip - 1] == 0
            and self._dots[self._line_tip][self._column_tip - 1] == 0
        ):
            actions.append(DirAction.LEFT)
        return actions

    def apply_action(self, a: DirAction):
        """
        Applies a given action to the state. It modifies the segments visited by the snake (v_seg and h_seg),
        the intersections (dots), and the tip of the snake.
        """
        # moving up
        if a == DirAction.UP:
            self._v_seg[self._line_tip][self._column_tip] = 1
            self._dots[self._line_tip + 1][self._column_tip] = 1
            self._line_tip += 1
        # moving down
        if a == DirAction.DOWN:
            self._v_seg[self._line_tip - 1][self._column_tip] = 1
            self._dots[self._line_tip - 1][self._column_tip] = 1
            self._line_tip -= 1
        # moving right
        if a == DirAction.RIGHT:
            self._h_seg[self._line_tip][self._column_tip] = 1
            self._dots[self._line_tip][self._column_tip + 1] = 1
            self._column_tip += 1
        # moving left
        if a == DirAction.LEFT:
            self._h_seg[self._line_tip][self._column_tip - 1] = 1
            self._dots[self._line_tip][self._column_tip - 1] = 1
            self._column_tip -= 1

    def has_tip_reached_goal(self):
        """
        Verifies whether the snake has reached the goal position. Note this is not a goal
        test. The goal test is performed by method is_solution, which uses has_tip_reached_goal
        as part of the verification.
        """
        return (
            self._line_tip == self._line_goal and self._column_tip == self._column_goal
        )

    def random_path(self):
        """
        Generates a path through a random walk, mostly used for debugging purposes. The random
        walk finishes as soon as the tip reaches the goal position or there are not more legal actions.
        """
        self.reset()

        actions = self.successors()
        while len(actions) > 0:
            a = random.randint(0, len(actions) - 1)
            self.apply_action(actions[a])
            # If the tip of random walk reached the goal, then stop
            if self.has_tip_reached_goal():
                return
            actions = self.successors()

    def is_solution(self):
        """
        Verifies whether the state's path represents a valid solution. This is performed by verifying the following
        (1) the tip of the snake is at the goal position
        (2) a bullet of color c1 cannot reach a bullet of color c2 through a BFS search.

        The BFS uses the cells (line and column) as states and verifies whether cells with a bullet of a given color
        can only reach (and be reached) by cells with bullets of the same color (or of the neutral color, denoted as zero in this implementation)
        """
        if not self.has_tip_reached_goal():
            return False

        non_visited_states = set()
        current_color = 0
        closed_bfs = np.zeros((self._lines, self._columns))
        for i in range(self._cells.shape[0]):
            for j in range(self._cells.shape[1]):
                non_visited_states.add((i, j))

        while len(non_visited_states) != 0:
            root = non_visited_states.pop()
            # If root of new BFS search was already visited, then go to the next state
            if closed_bfs[root[0]][root[1]] == 1:
                continue
            current_color = self._cells[root[0]][root[1]]

            open_bfs = deque()
            open_bfs.append(root)
            closed_bfs[root[0]][root[1]] = 1
            while len(open_bfs) != 0:
                # remove first state from queue
                state = open_bfs.popleft()
                children = self.__successor_bfs(state)
                for c in children:
                    # If c is a duplicate, then continue with the next child
                    if closed_bfs[c[0]][c[1]] == 1:
                        continue
                    # If c's color isn't neutral (zero) and it is different from current_color, then state isn't a soution
                    if (
                        current_color != 0
                        and self._cells[c[0]][c[1]] != 0
                        and self._cells[c[0]][c[1]] != current_color
                    ):
                        return False
                    # If current_color is neutral (zero) and c's color isn't, then attribute c's color to current_color
                    if self._cells[c[0]][c[1]] != 0:
                        current_color = self._cells[c[0]][c[1]]
                    # Add c to BFS's open list
                    open_bfs.append(c)
                    # mark state c as visited
                    closed_bfs[c[0]][c[1]] = 1
        return True
