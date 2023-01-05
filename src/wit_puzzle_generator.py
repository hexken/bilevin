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

    colors = range(1, args.colors + 1)
    goals = [(0, i) for i in range(args.width + 1)] + [
        (i, 0) for i in range(args.width + 1)
    ]
    dummy_problems = [
        Witness(
            f"Size: {args.width} {args.width}\nInit: 0 0\nGoal: {g[0]} {g[1]}\nColors: |"
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


def connected_components(state):
    """
    Compute the connected components of the grid, i.e. the regions separated by the path
    """
    visited = np.zeros((state.num_rows, state.num_cols))
    cell_states = [
        (i, j) for i, j in product(range(state.num_rows), range(state.num_cols))
    ]
    while len(cell_states) != 0:
        root = cell_states.pop()
        # If root of new BFS search was already visited, then go to the next state
        if visited[root] == 1:
            continue
        current_color = Color(state.cells[root])

        frontier = deque()
        frontier.append(root)
        visited[root] = 1
        while len(frontier) != 0:
            cell = frontier.popleft()

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
                if row + 1 < state.num_rows and state.h_segs[row + 1, col] == 0:
                    neighbors.append((row + 1, col))
                # move down
                if row > 0 and state.h_segs[row, col] == 0:
                    neighbors.append((row - 1, col))
                # move right
                if col + 1 < state.num_cols and state.v_segs[row, col + 1] == 0:
                    neighbors.append((row, col + 1))
                # move left
                if col > 0 and state.v_segs[row, col] == 0:
                    neighbors.append((row, col - 1))
                return neighbors

            neighbors = reachable_neighbors(state, cell)
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


if __name__ == "__main__":
    main()
