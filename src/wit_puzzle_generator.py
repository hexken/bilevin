import argparse
from collections import deque
from itertools import product
from pathlib import Path
import random

import numpy as np
import tqdm

from domains.witness import Witness, WitnessState
from enums import Color


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--output-path",
        type=lambda p: Path(p).absolute(),
        help="path to save problem instances",
    )

    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=4,
        help="width of puzzles to be generated",
    )

    parser.add_argument(
        "-n",
        "--num-problems",
        type=int,
        default=1000,
        help="number of training puzzles to be generated",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )

    parser.add_argument(
        "-c",
        "--num-colors",
        type=int,
        default=4,
        help="number of colors to use",
    )

    parser.add_argument(
        "-p",
        "--bullet-prob",
        dest="bullet_prob",
        default=0.6,
        help="probability of placing a bullet in each empty cell",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    # to.manual_seed(args.seed)

    colors = range(1, args.num_colors + 1)

    goals = (
        [(0, i) for i in range(args.width + 1)]
        + [(i, 0) for i in range(args.width + 1)]
        + [(args.width + 1, i) for i in range(args.width + 1)]
        + [(i, args.width + 1) for i in range(args.width + 1)]
    )
    goals.remove((0, 0))

    dummy_problems = [
        Witness(
            [
                [f"Size: {args.width} {args.width}"],
                [f"Init: 0 0"],
                [f"Goal: {g[0]} {g[1]}"],
                [f"Colors: |"],
            ]
        )
        for g in goals
    ]

    problems = set()

    with tqdm.tqdm(total=args.num_problems) as pbar:
        while len(problems) < args.num_problems:
            for wit in dummy_problems:
                # generate a path from start to goal
                state = wit.reset()
                while True:
                    actions = wit.actions_unpruned(state)
                    action = random.choice(actions)
                    state = wit.result(state, action)
                    if state.is_head_at_goal():
                        break
                # commpute the regions separated by the path
                regions = connected_components(state)

                min_num_regions = 2
                if args.width == 3:
                    min_num_regions = 2
                if args.width >= 4:
                    min_num_regions = 4
                if args.width >= 10:
                    min_num_regions = 5

                if len(regions) < min_num_regions:
                    continue

                # fill each region with a color


def connected_components(wit_state):
    """
    Compute the connected components of the grid, i.e. the regions separated by the path
    """
    visited = np.zeros((wit_state.num_rows, wit_state.num_cols))
    cell_states = [
        (i, j) for i, j in product(range(wit_state.num_rows), range(wit_state.num_cols))
    ]
    regions = []
    while len(cell_states) != 0:
        root = cell_states.pop()
        # If root of new BFS search was already visited, then go to the next state
        if visited[root] == 1:
            continue
        this_region = [root]
        frontier = deque()
        frontier.append(root)
        visited[root] = 1
        while len(frontier) != 0:
            cell_state = frontier.popleft()

            def reachable_neighbors(cell):
                """
                Successor function use in the Breadth-first search (BFS) performed to validate a solution.
                An adjacent cell c' is amongst the successors of cell c if there is no segment (v_seg or h_seg)
                separating cells c and c'.

                This method is meant to be called only from within GameState
                """
                neighbors = []
                row, col = cell
                # move up
                if row + 1 < wit_state.num_rows and wit_state.h_segs[row + 1, col] == 0:
                    neighbors.append((row + 1, col))
                # move down
                if row > 0 and wit_state.h_segs[row, col] == 0:
                    neighbors.append((row - 1, col))
                # move right
                if col + 1 < wit_state.num_cols and wit_state.v_segs[row, col + 1] == 0:
                    neighbors.append((row, col + 1))
                # move left
                if col > 0 and wit_state.v_segs[row, col] == 0:
                    neighbors.append((row, col - 1))
                return neighbors

            neighbors = reachable_neighbors(cell_state)
            for neighbor in neighbors:
                # If neighbor is a duplicate, then continue with the next child
                if visited[neighbor] == 1:
                    continue
                this_region.append(neighbor)
                frontier.append(neighbor)
                visited[neighbor] = 1
        regions.append(this_region)
    return regions


if __name__ == "__main__":
    main()
