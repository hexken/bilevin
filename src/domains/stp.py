from __future__ import annotations
from copy import deepcopy
from typing import Optional, Type

import numpy as np
import torch as to

from domains.domain import Domain, State
from enums import FourDir
from search.utils import SearchNode, Trajectory


def load_problemset(problemset: dict):
    width = problemset["width"]
    problems = []
    goal_tiles = np.arange(width**2).reshape(width, width)
    for p_dict in problemset["problems"]:
        init_tiles = np.array(p_dict["init_tiles"])
        problem = SlidingTilePuzzle(
            init_tiles=init_tiles, goal_tiles=np.array(goal_tiles)
        )
        problems.append((p_dict["id"], problem))

    num_actions = problems[0][1].num_actions
    in_channels = problems[0][1].in_channels
    state_t_width = problems[0][1].state_width
    double_backward = False

    return problems, num_actions, in_channels, state_t_width, double_backward


class SlidingTilePuzzleState(State):
    def __init__(
        self,
        tiles: np.ndarray,
        blank_row: int,
        blank_col: int,
    ):
        self.tiles = tiles
        self.blank_row = blank_row
        self.blank_col = blank_col

    def __repr__(self) -> str:
        width = self.tiles.shape[0]
        return f" {np.array2string( self.tiles.flatten(), separator=' ' , max_line_width=width * 2)[1:-1]}"

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> int:
        return self.tiles.tobytes().__hash__()

    def __eq__(self, other) -> bool:
        raise NotImplementedError
        return np.array_equal(self.tiles, other.tiles)


class SlidingTilePuzzle(Domain):
    def __init__(self, init_tiles: np.ndarray, goal_tiles: np.ndarray):
        self.forward = True

        self.width = init_tiles.shape[0]
        self.num_tiles = self.width**2

        blank_pos = np.where(init_tiles == 0)
        self.blank_row = blank_pos[0].item()
        self.blank_col = blank_pos[1].item()

        self.initial_state = SlidingTilePuzzleState(
            init_tiles, self.blank_row, self.blank_col
        )
        self.initial_state_t = self.state_tensor(self.initial_state)

        width_indices = np.arange(self.width)
        self._row_indices = np.repeat(width_indices, self.width)
        self._col_indices = np.tile(width_indices, self.width)

        self.goal_tiles = goal_tiles

        self.visited = {}

    def reset(self) -> SlidingTilePuzzleState:
        self.visited = {}
        return self.initial_state

    def update(self, node: SearchNode):
        self.visited[node.state.__hash__()] = node

    @property
    def state_width(self) -> int:
        return self.width

    @property
    def num_actions(cls) -> int:
        return 4

    @property
    def in_channels(self) -> int:
        return self.num_tiles

    def state_tensor(
        self,
        state: SlidingTilePuzzleState,
    ) -> to.Tensor | tuple[to.Tensor, to.Tensor]:
        arr = to.zeros((self.num_tiles, self.width, self.width))
        arr[
            state.tiles.reshape(-1),
            self._row_indices,
            self._col_indices,
        ] = 1

        if self.forward:
            return arr
        else:
            return to.stack(
                (arr, self.initial_state_t)  # type:ignore
            )

    def reverse_action(self, action: FourDir):
        if action == FourDir.UP:
            return FourDir.DOWN
        elif action == FourDir.DOWN:
            return FourDir.UP
        elif action == FourDir.LEFT:
            return FourDir.RIGHT
        elif action == FourDir.RIGHT:
            return FourDir.LEFT

    def backward_problem(self) -> SlidingTilePuzzle:
        assert self.forward
        problem = deepcopy(self)
        problem.forward = False
        forward_goal = problem.goal
        problem.goal = problem.initial_state.tiles

        problem.initial_state = SlidingTilePuzzleState(
            forward_goal,
            0,
            0,
            problem.width,
        )

        return problem

    def actions(
        self, parent_action: FourDir, state: SlidingTilePuzzleState
    ) -> list[FourDir]:
        return self.actions_unpruned(state)

    def actions_unpruned(self, state: SlidingTilePuzzleState):
        actions = []
        blank_row = state.blank_pos[0]
        blank_col = state.blank_pos[1]

        if blank_col != state.width - 1:
            actions.append(FourDir.RIGHT)

        if blank_row != 0:
            actions.append(FourDir.UP)

        if blank_col != 0:
            actions.append(FourDir.LEFT)

        if blank_row != state.width - 1:
            actions.append(FourDir.DOWN)

        return actions

    def result(
        self, state: SlidingTilePuzzleState, action: FourDir
    ) -> SlidingTilePuzzleState:
        blank_pos = state.blank_pos
        blank_row = state.blank_pos[0]
        blank_col = state.blank_pos[1]

        new_state = deepcopy(state)

        if action == FourDir.UP:
            new_state.tiles[blank_pos], new_state.tiles[blank_row - 1, blank_col] = (
                new_state.tiles[blank_row - 1, blank_col],
                new_state.tiles[blank_pos],
            )
            new_state.blank_pos = (blank_row - 1, blank_col)

        elif action == FourDir.DOWN:
            new_state.tiles[blank_pos], new_state.tiles[blank_row + 1, blank_col] = (
                new_state.tiles[blank_row + 1, blank_col],
                new_state.tiles[blank_pos],
            )
            new_state.blank_pos = (blank_row + 1, blank_col)

        elif action == FourDir.RIGHT:
            new_state.tiles[blank_pos], new_state.tiles[blank_row, blank_col + 1] = (
                new_state.tiles[blank_row, blank_col + 1],
                new_state.tiles[blank_pos],
            )
            new_state.blank_pos = (blank_row, blank_col + 1)

        elif action == FourDir.LEFT:
            new_state.tiles[blank_pos], new_state.tiles[blank_row, blank_col - 1] = (
                new_state.tiles[blank_row, blank_col - 1],
                new_state.tiles[blank_pos],
            )
            new_state.blank_pos = (blank_row, blank_col - 1)

        return new_state

    def is_goal(self, state: SlidingTilePuzzleState):
        return np.array_equal(state.tiles, self.goal)

    def get_merged_trajectory(
        self,
        dir1_common: SearchNode,
        dir2_common: SearchNode,
        node_type: Type[SearchNode],
        num_expanded: int,
    ):
        """
        Returns a new trajectory going from dir1_start to dir2_start, passing through
        merge(dir1_common, dir2_common).
        """
        dir1_node = dir1_common
        parent_dir2_node = dir2_common.parent
        parent_dir2_action = dir2_common.parent_action
        while parent_dir2_node:
            new_state = deepcopy(parent_dir2_node.state)

            new_dir1_node = node_type(
                state=new_state,
                parent=dir1_node,
                parent_action=self.reverse_action(parent_dir2_action),
                g_cost=dir1_node.g_cost + 1,
            )
            dir1_node = new_dir1_node
            parent_dir2_action = parent_dir2_node.parent_action
            parent_dir2_node = parent_dir2_node.parent

        return Trajectory(self, dir1_node, num_expanded)

    def try_make_solution(
        self,
        node: SearchNode,
        other_problem: SlidingTilePuzzle,
        num_expanded: int,
    ) -> Optional[tuple[Trajectory, Trajectory]]:
        """
        Returns a trajectory if state is a solution to this problem, None otherwise.
        """
        hsh = node.state.__hash__()
        if hsh in other_problem.visited:
            other_node = other_problem.visited[hsh]
            if self.forward:
                f_common_node = node
                b_common_node = other_node
            else:
                f_common_node = other_node
                b_common_node = node

            f_traj = self.get_merged_trajectory(
                f_common_node, b_common_node, type(node), num_expanded
            )
            b_traj = self.get_merged_trajectory(
                b_common_node, f_common_node, type(node), num_expanded
            )

            return (f_traj, b_traj)
        else:
            return None
