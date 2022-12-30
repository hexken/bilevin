from collections import defaultdict, deque
from copy import deepcopy
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import torch as to

from domains.domain import Domain, State
from enums import Color, FourDir
from src.search.utils import SearchNode


class WitnessState(State):
    def __init__(
        self,
        start_y: int,
        start_x: int,
        goal_y: int,
        goal_x: int,
        cells: np.ndarray,
        num_rows: int,
        num_cols: int,
        max_rows: int,
        max_cols: int,
    ):
        self.start_y = start_y
        self.start_x = start_x
        self.head_y = start_y
        self.head_x = start_x

        self.goal_x = goal_y
        self.goal_y = goal_x

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cells = cells

        self.max_rows = max_rows
        self.max_cols = max_cols

        self.dots = np.zeros((num_rows + 1, num_cols + 1))
        self.dots[self.head_y][self.head_x] = 1

        self.v_segs = np.zeros((num_rows, num_cols + 1))
        self.h_segs = np.zeros((num_rows + 1, num_cols))

    def is_head_at_goal(self):
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
        # todo I removed the cells check in hash and eq, since they should all be the same for a
        # given problem. Double check
        return hash((self.v_segs.tobytes(), self.h_segs.tobytes()))

    def __eq__(self, other):
        return (
            np.array_equal(self.v_segs, other.v_segs)
            and np.array_equal(self.h_segs, other.h_segs)
            and np.array_equal(self.dots, other.dots)
        )

    def plot(self, filename=None):
        """
        This method plots the state. Several features in this method are hard-coded and might
        need adjustment as one changes the size of the puzzle. For example, the size of the figure is set to be fixed
        to [5, 5] (see below).
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        # fig.patch.set_facecolor((1, 1, 1))

        # draw vertical lines of the grid
        for y in range(self.dots.shape[1]):
            ax.plot([y, y], [0, self.cells.shape[0]], str(Color.BLACK))
        # draw horizontal lines of the grid
        for x in range(self.dots.shape[0]):
            ax.plot([0, self.cells.shape[1]], [x, x], str(Color.BLACK))

        # scale the axis area to fill the whole figure
        ax.set_position([0, 0, 1, 1])

        ax.set_axis_off()

        ax.set_xlim(-1, np.max(self.dots.shape))
        ax.set_ylim(-1, np.max(self.dots.shape))

        # Draw the vertical segments of the path
        for i in range(self.v_segs.shape[0]):
            for j in range(self.v_segs.shape[1]):
                if self.v_segs[i][j] == 1:
                    ax.plot([j, j], [i, i + 1], str(Color.RED), linewidth=5)

        # Draw the horizontal segments of the path
        for i in range(self.h_segs.shape[0]):
            for j in range(self.h_segs.shape[1]):
                if self.h_segs[i][j] == 1:
                    ax.plot([j, j + 1], [i, i], str(Color.RED), linewidth=5)

        # Draw the separable bullets according to the values in self.cells and Color enum type
        offset = 0.5
        color_strings = Color.str_values()[1:]
        for i in range(self.cells.shape[0]):
            for j in range(self.cells.shape[1]):
                if self.cells[i][j] != 0:
                    ax.plot(
                        j + offset,
                        i + offset,
                        "o",
                        markersize=15,
                        markeredgecolor=(0, 0, 0),
                        markerfacecolor=color_strings[int(self.cells[i][j] - 1)],
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
                        markerfacecolor=str(Color.RED),
                        markeredgewidth=0,
                    )
                else:
                    ax.plot(
                        j,
                        i,
                        "o",
                        markersize=10,
                        markeredgecolor=(0, 0, 0),
                        markerfacecolor=str(Color.BLACK),
                        markeredgewidth=0,
                    )

        # Draw the entrance of the puzzle in red as it is always on the state's path
        ax.plot(
            self.start_x - 0.15,
            self.start_y,
            ">",
            markersize=10,
            markeredgecolor=(0, 0, 0),
            markerfacecolor=str(Color.RED),
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
                markerfacecolor=str(Color.BLACK),
                markeredgewidth=0,
            )
        else:
            ax.plot(
                self.goal_x + column_exit_offset,
                self.goal_y + row_exit_offset,
                exit_symbol,
                markersize=10,
                markeredgecolor=(0, 0, 0),
                markerfacecolor=str(Color.RED),
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
        puzzle,
        max_rows=11,
        max_cols=11,
    ):
        assert max_rows == max_cols
        self.max_rows = max_rows
        self.max_cols = max_cols

        values = puzzle[0].replace("Size: ", "").split(" ")
        self.num_rows = int(values[0])
        self.num_cols = int(values[1])

        values = puzzle[1].replace("Init: ", "").split(" ")
        self.start_y = int(values[0])
        self.start_x = int(values[1])

        values = puzzle[2].replace("Goal: ", "").split(" ")
        self.goal_y = int(values[0])
        self.goal_x = int(values[1])

        self.cells = np.zeros((self.num_rows, self.num_cols), dtype=np.int32)
        values = puzzle[3].replace("Colors: |", "").split("|")
        for t in values:
            numbers = t.split(" ")
            self.cells[int(numbers[0])][int(numbers[1])] = int(numbers[2])

        self.initial_state = WitnessState(
            self.start_y,
            self.start_x,
            self.goal_y,
            self.goal_x,
            self.cells,
            self.num_rows,
            self.num_cols,
            self.max_rows,
            self.max_cols,
        )

        self.heads = {}

    def reset(self):
        return self.initial_state  # todo might need deeopcopy, check mutability reqs

    def update(self, node: SearchNode):
        head = (node.state.head_x, node.state.head_y)
        if head in self.heads:
            self.heads[head].append(node)
        else:
            self.heads[head] = [node]

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

    def actions(self, parent_action: FourDir, state: WitnessState):
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
            and state.head_y >= 1
            and state.v_segs[state.head_y - 1][state.head_x] == 0
            and state.dots[state.head_y - 1][state.head_x] == 0
        ):
            actions.append(FourDir.DOWN)
        # moving right
        if (
            parent_action != FourDir.LEFT
            and state.head_x + 1 < state.dots.shape[1]
            and state.h_segs[state.head_y][state.head_x] == 0
            and state.dots[state.head_y][state.head_x + 1] == 0
        ):
            actions.append(FourDir.RIGHT)
        # moving left
        if (
            parent_action != FourDir.RIGHT
            and state.head_x >= 1
            and state.h_segs[state.head_y][state.head_x - 1] == 0
            and state.dots[state.head_y][state.head_x - 1] == 0
        ):
            actions.append(FourDir.LEFT)

        return actions

    def actions_unpruned(self, state: WitnessState):
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
            state.head_y + 1 < state.dots.shape[0]
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
            state.head_x + 1 < state.dots.shape[1]
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

    def result(self, state: WitnessState, action: FourDir):
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

    def is_goal(self, state: WitnessState):
        """
        Verifies whether the state's path represents a valid solution. This is performed by verifying the following
        (1) the tip of the snake is at the goal position
        (2) a bullet of color c1 cannot reach a bullet of color c2 through a BFS search.

        The BFS uses the cells (line and column) as states and verifies whether cells with a bullet of a given color
        can only reach (and be reached) by cells with bullets of the same color (or of the neutral color, denoted as zero in this implementation)
        """
        if not state.is_head_at_goal():
            return False

        # state.plot()
        current_color = Color.NEUTRAL
        visited = np.zeros((self.num_rows, self.num_cols))
        cells = [(i, j) for i, j in product(range(self.num_rows), range(self.num_cols))]

        while len(cells) != 0:
            root = cells.pop()
            # If root of new BFS search was already visited, then go to the next state
            if visited[root] == 1:
                continue
            current_color = Color(state.cells[root])

            frontier = deque()
            frontier.append(root)
            visited[root] = 1
            while len(frontier) != 0:
                # remove first cell (state) from queue
                cell = frontier.popleft()

                def reachable_neighbors(self, cell):
                    """
                    Successor function use in the Breadth-first search (BFS) performed to validate a solution.
                    An adjacent cell c' is amongst the successors of cell c if there is no segment (v_seg or h_seg)
                    separating cells c and c'.

                    This method is meant to be called only from within GameState
                    """
                    neighbors = []
                    row, col = cell
                    # move up
                    if row + 1 < self.num_rows and state.h_segs[row + 1][col] == 0:
                        neighbors.append((row + 1, col))
                    # move down
                    if row > 0 and state.h_segs[row][col] == 0:
                        neighbors.append((row - 1, col))
                    # move right
                    if col + 1 < self.num_cols and state.v_segs[row][col + 1] == 0:
                        neighbors.append((row, col + 1))
                    # move left
                    if col > 0 and state.v_segs[row][col] == 0:
                        neighbors.append((row, col - 1))
                    return neighbors

                neighbors = reachable_neighbors(self, cell)
                for neighbor in neighbors:
                    # If neighbor is a duplicate, then continue with the next child
                    if visited[neighbor] == 1:
                        continue
                    # If neighbor's color isn't neutral (zero) and it is different from current_color, then state isn't a soution
                    if (
                        current_color != Color.NEUTRAL
                        and Color(state.cells[neighbor]) != Color.NEUTRAL
                        and Color(state.cells[neighbor]) != current_color
                    ):
                        return False
                    # If current_color is neutral (zero) and neighbor's color isn't, then attribute c's color to current_color
                    if current_color == Color.NEUTRAL:
                        current_color = Color(state.cells[neighbor])
                    # Add c to BFS's open list
                    frontier.append(neighbor)
                    # mark state c as visited
                    visited[neighbor] = 1
        return True

    def is_bidirectional_goal(self, node, other_problem):
        state = node.state
        head_dot = (state.head_x, state.head_y)
        if head_dot not in other_problem.heads:
            return
        for other_node in other_problem.heads[head_dot]:
            merged_state = deepcopy(state)
            other_state = other_node.state
            merged_state.dots += other_state.dots
            merged_state.v_segs += other_state.v_segs
            merged_state.h_segs += other_state.h_segs
            merged_state.head_x = other_state.goal_x
            merged_state.head_y = other_state.goal_y

    def backward_problem(self):
        domain = deepcopy(self)

        domain.goal_x = self.start_x
        domain.goal_y = self.start_y
        domain.start_x = self.goal_x
        domain.start_y = self.goal_y

        domain.initial_state.dots[self.start_y, self.start_x] = 0
        domain.initial_state.dots[self.goal_y, self.goal_x] = 1
        domain.initial_state.goal_x = self.start_x
        domain.initial_state.goal_y = self.start_y
        domain.initial_state.start_x = self.goal_x
        domain.initial_state.start_y = self.goal_y

        return domain

    def __repr__(self):
        return self.initial_state.__repr__()

    def plot(self, filename=None):
        self.initial_state.plot(filename)
