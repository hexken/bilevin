from __future__ import annotations
from collections import deque
from typing import Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
import torch as to

from domains.domain import Domain, State
from enums import Color, ActionDir
if TYPE_CHECKING:
    from search.node import SearchNode


class WitnessState(State):
    """
    A Witness State.
    Note that start/head/goal row/col's refer to the grid, not cell locations
    Note that head_row and head_col are only consistent with grids/segs upon initialization if they're 0
    """

    def __init__(
        self,
        width: int,
        head_init_row,
        head_init_col,
        partial_init=False,
    ):
        self.head_row = head_init_row
        self.head_col = head_init_col
        self.width = width  # width of cells

        if not partial_init:
            self.grid = np.zeros((self.width + 1, self.width + 1))
            self.grid[self.head_row, self.head_col] = 1

            self.v_segs = np.zeros((self.width, self.width + 1))
            self.h_segs = np.zeros((self.width + 1, self.width))

    def __hash__(self) -> int:
        """
        Note that these hash and eq implementations imply the states are generated
        from the same problem
        """
        return (
            self.h_segs.tobytes(),
            self.v_segs.tobytes(),
        ).__hash__()

    def __eq__(self, other) -> bool:
        return (
            np.array_equal(self.grid, other.grid)
            and np.array_equal(self.h_segs, other.h_segs)
            and np.array_equal(self.v_segs, other.v_segs)
        )


class Witness(Domain):
    def __init__(
        self,
        puzzle: str,
        initial_state: WitnessState,
        goal_row: int,
        goal_col: int,
        markers: list[tuple[int, int, int]],
        forward: bool = True,
    ):
        super().__init__(forward=forward)
        self.puzzle = puzzle
        self.initial_state: WitnessState = initial_state
        self.goal_row = goal_row
        self.goal_col = goal_col
        self.markers = markers

    def reset(self) -> State:
        self.goal_state_t = None
        if self.puzzle == "triangles":
            self.max_num_colors = 3  # blue, red, green
        elif self.puzzle == "colors":
            self.max_num_colors = 4  # blue, red, green, cyan
        else:
            raise ValueError("Invalid puzzle type")

        self.width = self.initial_state.width
        self.start_row = self.initial_state.head_row
        self.start_col = self.initial_state.head_col

        self.cells = np.zeros((self.width, self.width), dtype=np.int32)

        for cell in self.markers:
            i, j, color = cell
            self.cells[i, j] = color

        if self.puzzle == "colors":
            self.is_goal = self.colors_is_goal
            self._colored_idxs = [
                (i, j)
                for i in range(self.width)
                for j in range(self.width)
                if self.cells[i, j] != 0
            ]
        elif self.puzzle == "triangles":
            self.is_goal = self.triangles_is_goal
            self._triangles = [
                (self.cells[i, j], i, j)
                for i in range(self.width)
                for j in range(self.width)
                if self.cells[i, j] != 0
            ]
        else:
            raise ValueError("Invalid puzzle type")

        return self._reset()

    def update(self, node: SearchNode):
        state: WitnessState = node.state
        head = (state.head_row, state.head_col)
        if head in self.aux_closed:
            self.aux_closed[head].append(node)
        else:
            self.aux_closed[head] = [node]

    @property
    def state_t_width(self) -> int:
        return self.width + 1

    @property
    def state_t_depth(self) -> int:
        return 1

    @property
    def num_actions(self) -> int:
        return 4

    @property
    def in_channels(self) -> int:
        """
        The max num of colors any problem in a particular problem set may have.
        """
        return self.max_num_colors + 5

    def is_goal(self, state: WitnessState) -> bool:
        """
        Checks if the current state is a goal state.
        """
        raise NotImplementedError

    def state_tensor(self, state: WitnessState) -> to.Tensor:
        """
        Generates an image representation for the puzzle. one channel for each color; one channel with 1's
        where is "open" in the grid (this allows learning systems to work with a fixed image size defined
        by max_lines and max_columns); one channel for the current path (cells occupied by the snake);
        one channel for the tip of the snake; one channel for the exit of the puzzle; one channel for the
        entrance of the snake. In total there are 9 different channels.

        Each channel is a matrix with zeros and ones. The image returned is a 3-dimensional numpy array.
        """

        # defining the 3-dimnesional array that will be filled with the puzzle's information
        arr = np.zeros(
            (self.in_channels, self.state_t_width, self.state_t_width), dtype=np.float32
        )

        for i in range(self.width):
            for j in range(self.width):
                color = self.cells[i, j]
                if color != 0:
                    arr[
                        color - 1, i, j
                    ] = 1  # -1 because we don't encode the neutral color

        channel_number = self.max_num_colors
        # channels for the current path
        # vsegs
        for i in range(self.width):
            for j in range(self.width + 1):
                if state.v_segs[i, j] == 1:
                    arr[channel_number, i, j] = 1

        channel_number += 1
        # hsegs
        for i in range(self.width + 1):
            for j in range(self.width):
                if state.h_segs[i, j] == 1:
                    arr[channel_number, i, j] = 1

        # channel with the tip of the snake
        channel_number += 1
        arr[channel_number, state.head_row, state.head_col] = 1

        # channel for the exit of the puzzle
        channel_number += 1
        arr[channel_number, self.goal_row, self.goal_col] = 1

        # channel for the entrance of the puzzle
        channel_number += 1
        arr[channel_number, self.start_row, self.start_col] = 1

        return to.from_numpy(arr)

    def reverse_action(self, action: ActionDir) -> ActionDir:
        if action == ActionDir.UP:
            return ActionDir.DOWN
        elif action == ActionDir.DOWN:
            return ActionDir.UP
        elif action == ActionDir.LEFT:
            return ActionDir.RIGHT
        elif action == ActionDir.RIGHT:
            return ActionDir.LEFT

    def _actions(self, parent_action: ActionDir, state: WitnessState) -> list[ActionDir]:
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
        # moving up
        if (
            parent_action != ActionDir.DOWN
            and state.head_row + 1 < state.grid.shape[0]
            and state.grid[state.head_row + 1, state.head_col] == 0
        ):
            actions.append(ActionDir.UP)
        # moving down
        if (
            parent_action != ActionDir.UP
            and state.head_row >= 1
            and state.grid[state.head_row - 1, state.head_col] == 0
        ):
            actions.append(ActionDir.DOWN)
        # moving right
        if (
            parent_action != ActionDir.LEFT
            and state.head_col + 1 < state.grid.shape[1]
            and state.grid[state.head_row, state.head_col + 1] == 0
        ):
            actions.append(ActionDir.RIGHT)
        # moving left
        if (
            parent_action != ActionDir.RIGHT
            and state.head_col >= 1
            and state.grid[state.head_row, state.head_col - 1] == 0
        ):
            actions.append(ActionDir.LEFT)

        return actions

    def _actions_unpruned(self, state: WitnessState) -> list[ActionDir]:
        actions = []
        # should return the same actions as pruned
        # moving up
        if (
            state.head_row + 1 < state.grid.shape[0]
            and state.grid[state.head_row + 1, state.head_col] == 0
        ):
            actions.append(ActionDir.UP)
        # moving down
        if state.head_row >= 1 and state.grid[state.head_row - 1, state.head_col] == 0:
            actions.append(ActionDir.DOWN)
        # moving right
        if (
            state.head_col + 1 < state.grid.shape[1]
            and state.grid[state.head_row, state.head_col + 1] == 0
        ):
            actions.append(ActionDir.RIGHT)
        # moving left
        if state.head_col >= 1 and state.grid[state.head_row, state.head_col - 1] == 0:
            actions.append(ActionDir.LEFT)

        return actions

    def result(self, state: WitnessState, action: ActionDir) -> WitnessState:
        """
        Applies a given action to the state. It modifies the segments visited by the snake (v_seg and h_seg),
        the intersections (grid), and the tip of the snake.
        """
        new_state = WitnessState(
            self.width, state.head_row, state.head_col, partial_init=True
        )
        new_state.grid = np.array(state.grid)
        new_state.v_segs = np.array(state.v_segs)
        new_state.h_segs = np.array(state.h_segs)

        # moving up
        if action == ActionDir.UP:
            new_state.v_segs[new_state.head_row, new_state.head_col] = 1
            new_state.grid[new_state.head_row + 1, new_state.head_col] = 1
            new_state.head_row += 1
        # moving down
        elif action == ActionDir.DOWN:
            new_state.v_segs[new_state.head_row - 1, new_state.head_col] = 1
            new_state.grid[new_state.head_row - 1, new_state.head_col] = 1
            new_state.head_row -= 1
        # moving right
        elif action == ActionDir.RIGHT:
            new_state.h_segs[new_state.head_row, new_state.head_col] = 1
            new_state.grid[new_state.head_row, new_state.head_col + 1] = 1
            new_state.head_col += 1
        # moving left
        elif action == ActionDir.LEFT:
            new_state.h_segs[new_state.head_row, new_state.head_col - 1] = 1
            new_state.grid[new_state.head_row, new_state.head_col - 1] = 1
            new_state.head_col -= 1

        return new_state

    def is_head_at_goal(self, state: WitnessState) -> bool:
        return self.goal_row == state.head_row and self.goal_col == state.head_col

    def triangles_is_goal(self, state: WitnessState) -> bool:
        if not self.is_head_at_goal(state):
            return False

        for n, i, j in self._triangles:
            if (
                state.v_segs[i, j]
                + state.v_segs[i, j + 1]
                + state.h_segs[i, j]
                + state.h_segs[i + 1, j]
                != n
            ):
                return False
        return True

    def colors_is_goal(self, state: WitnessState) -> bool:
        """
        Verifies whether the state's path represents a valid solution. This is performed by verifying the following
        (1) the tip of the snake is at the goal position
        (2) a bullet of color c1 cannot reach a bullet of color c2 through a BFS search.

        The BFS uses the cells (line and column) as states and verifies whether cells with a bullet of a given color
        can only reach (and be reached) by cells with bullets of the same color (or of the neutral color, denoted as zero in this implementation)
        """
        if not self.is_head_at_goal(state):
            return False

        reached = set()

        for root in self._colored_idxs:
            # If root of new BFS search was already visited, then go to the next state
            if root in reached:
                continue
            current_color = self.cells[root]

            frontier = deque()
            frontier.append(root)
            reached.add(root)
            while len(frontier) != 0:
                cell = frontier.popleft()

                def reachable_neighbors(self, cell) -> list[tuple[int, int]]:
                    """
                     Breadth-first search (BFS) performed to validate a solution.
                    An adjacent cell c' is amongst the successors of cell c if there is no segment (v_seg or h_seg)
                    separating cells c and c'.

                    """
                    neighbors = []
                    row, col = cell
                    # move up
                    if row + 1 < self.width and state.h_segs[row + 1, col] == 0:
                        neighbors.append((row + 1, col))
                    # move down
                    if row > 0 and state.h_segs[row, col] == 0:
                        neighbors.append((row - 1, col))
                    # move right
                    if col + 1 < self.width and state.v_segs[row, col + 1] == 0:
                        neighbors.append((row, col + 1))
                    # move left
                    if col > 0 and state.v_segs[row, col] == 0:
                        neighbors.append((row, col - 1))
                    return neighbors

                neighbors = reachable_neighbors(self, cell)
                for neighbor in neighbors:
                    if neighbor in reached:
                        continue
                    if (
                        self.cells[neighbor] != 0
                        and self.cells[neighbor] != current_color
                    ):
                        return False
                    frontier.append(neighbor)
                    reached.add(neighbor)
        return True

    def backward_domain(self) -> Witness:
        """
        Should only be called on a fresh domain (no calls to update). Reverses a witness problem by
        reversing the head and goal (and updating grid to be consistent with this change).
        """
        assert self.forward

        goal_row = self.initial_state.head_row
        goal_col = self.initial_state.head_col
        initial_state = WitnessState(self.width, self.goal_row, self.goal_col)
        domain = Witness(
            self.puzzle,
            initial_state,
            goal_row,
            goal_col,
            self.markers,
            forward=False,
        )

        return domain

    def plot(self, state: Optional[WitnessState] = None, filename=None):
        """
        This method plots the state. Several features in this method are hard-coded and might
        need adjustment as one changes the size of the puzzle. For example, the size of the figure is set to be fixed
        to [5, 5] (see below).
        """
        if not state:
            state = self.initial_state

        ax: plt.Axes
        _, ax = plt.subplots(figsize=(5, 5))
        # fig.patch.set_facecolor((1, 1, 1))

        # draw vertical lines of the grid
        for y in range(state.grid.shape[1]):
            ax.plot([y, y], [0, self.width], str(Color.BLACK))
        # draw horizontal lines of the grid
        for x in range(state.grid.shape[0]):
            ax.plot([0, self.width], [x, x], str(Color.BLACK))

        # scale the axis area to fill the whole figure
        ax.set_position([0, 0, 1, 1])

        ax.set_axis_off()

        ax.set_xlim(-1, np.max(state.grid.shape))
        ax.set_ylim(-1, np.max(state.grid.shape))

        # Draw the vertical segments of the path
        for i in range(state.v_segs.shape[0]):
            for j in range(state.v_segs.shape[1]):
                if state.v_segs[i, j] == 1:
                    ax.plot([j, j], [i, i + 1], str(Color.RED), linewidth=5)

        # Draw the horizontal segments of the path
        for i in range(state.h_segs.shape[0]):
            for j in range(state.h_segs.shape[1]):
                if state.h_segs[i, j] == 1:
                    ax.plot([j, j + 1], [i, i], str(Color.RED), linewidth=5)

        # Draw the separable bullets according to the values in self.cells and Color enum type
        offset = 0.5
        color_strings = Color.str_values()[1:]
        for i in range(self.width):
            for j in range(self.width):
                if self.cells[i, j] != 0:
                    ax.plot(
                        j + offset,
                        i + offset,
                        "o",
                        markersize=15,
                        markeredgecolor=(0, 0, 0),
                        markerfacecolor=color_strings[int(self.cells[i, j] - 1)],
                        markeredgewidth=2,
                    )

        # Draw the intersection of lines: red for an intersection that belongs to a path and black otherwise
        for i in range(state.grid.shape[0]):
            for j in range(state.grid.shape[1]):
                if state.grid[i, j] != 0:
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
            self.start_col - 0.15,
            self.start_row,
            ">",
            markersize=10,
            markeredgecolor=(0, 0, 0),
            markerfacecolor=str(Color.RED),
            markeredgewidth=0,
        )

        column_exit_offset = 0
        row_exit_offset = 0

        if self.goal_col == self.width:
            column_exit_offset = 0.15
            exit_symbol = ">"
        elif self.goal_col == 0:
            column_exit_offset = -0.15
            exit_symbol = "<"
        elif self.goal_row == self.width:
            row_exit_offset = 0.15
            exit_symbol = "^"
        else:
            row_exit_offset = -0.15
            exit_symbol = "v"
        # Draw the exit of the puzzle: red if it is on a path, black otherwise
        if state.grid[self.goal_row, self.goal_col] == 0:
            ax.plot(
                self.goal_col + column_exit_offset,
                self.goal_row + row_exit_offset,
                exit_symbol,
                markersize=10,
                markeredgecolor=(0, 0, 0),
                markerfacecolor=str(Color.BLACK),
                markeredgewidth=0,
            )
        else:
            ax.plot(
                self.goal_col + column_exit_offset,
                self.goal_row + row_exit_offset,
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

    def is_merge_goal(self, state, other_domain) -> Optional[SearchNode]:
        head_dot = (state.head_row, state.head_col)
        if head_dot not in other_domain.aux_closed:
            return None
        for other_node in other_domain.aux_closed[head_dot]:
            other_state = other_node.state

            merged_state = WitnessState(
                self.width,
                other_domain.start_row,
                other_domain.start_col,
                partial_init=False,
            )

            merged_state.grid = state.grid + other_state.grid
            # make sure snakes do not overlap any where but the head
            merged_state.grid[head_dot] = 1
            if (merged_state.grid > 1.5).any():
                return None

            merged_state.v_segs = state.v_segs + other_state.v_segs
            merged_state.h_segs = state.h_segs + other_state.h_segs
            if self.is_goal(merged_state):
                return other_node
        return None

    def get_merge_state(self, dir1_state, dir2_parent_state, action) -> State:
        return self.result(dir1_state, action)
