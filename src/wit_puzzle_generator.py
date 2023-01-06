import argparse
from collections import deque
from itertools import product
from pathlib import Path
from copy import copy
import random

import numpy as np
from tqdm import tqdm

from domains.witness import Witness


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

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    goals = (
        [(0, i) for i in range(args.width + 1)]
        + [(i, 0) for i in range(args.width + 1)]
        + [(args.width, i) for i in range(args.width + 1)]
        + [(i, args.width) for i in range(args.width + 1)]
    )
    goals.remove((0, 0))

    partial_specs = [
        [
            f"Size: {args.width} {args.width}",
            f"Init: 0 0",
            f"Goal: {g[0]} {g[1]}",
        ]
        for g in goals
    ]

    colors_prefix = "Colors: |"

    problem_specs = set()

    with tqdm(total=args.num_problems) as pbar:
        while len(problem_specs) < args.num_problems:
            head_at_goal = False
            partial_spec = random.choice(partial_specs)
            wit = Witness(partial_spec)
            # generate a path from start to goal
            state = wit.reset()
            while actions := wit.actions_unpruned(state):
                action = random.choice(actions)
                state = wit.result(state, action)
                if wit.is_head_at_goal(state):
                    head_at_goal = True
                    break

            if not head_at_goal:
                continue

            regions = connected_components(wit, state)

            min_num_regions = 2
            if args.width == 3:
                min_num_regions = 2
            if args.width >= 4:
                min_num_regions = 4
            if args.width == 10:
                min_num_regions = 5

            # todo should we allow regions to have the same color?
            if len(regions) < min_num_regions or len(regions) > args.num_colors:
                continue

            # fill each region with a color
            colors = random.sample(range(1, args.num_colors + 1), len(regions))
            color_str = colors_prefix
            for i, region in enumerate(regions):
                region_arr = np.array(sorted(region))
                region_mask = np.random.rand(len(region_arr)) < args.bullet_prob
                region_arr = region_arr[region_mask]
                if len(region_arr):
                    color_str += "|".join(
                        f"{row} {col} {colors[-1]}" for row, col in region_arr
                    )
                    if i < len(regions) - 1:
                        color_str += "|"
                    colors.pop()

            # Only add if at least two colors were used
            num_colors_used = len(regions) - len(colors)
            if num_colors_used < 2:
                continue

            # todo should we consider all problems with the same path as duplicates? Might lead to
            # poorer generalization
            partial_spec = copy(partial_spec)
            partial_spec.append(f"Num Colors: {num_colors_used}")
            partial_spec.append(color_str)
            formatted_spec = "%s\n" % "\n".join(partial_spec)
            if formatted_spec in problem_specs:
                continue
            else:
                problem_specs.add(formatted_spec)
                pbar.update()

    with args.output_path.open("w") as f:
        for spec in problem_specs:
            f.write(f"{spec}\n")


def connected_components(wit, wit_state):
    """
    Compute the connected components of the grid, i.e. the regions separated by the path
    """
    visited = np.zeros((wit.num_rows, wit.num_cols))
    cell_states = [(i, j) for i, j in product(range(wit.num_rows), range(wit.num_cols))]
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
                neighbors = []
                row, col = cell
                # move up
                if row + 1 < wit.num_rows and wit_state.h_segs[row + 1, col] == 0:
                    neighbors.append((row + 1, col))
                # move down
                if row > 0 and wit_state.h_segs[row, col] == 0:
                    neighbors.append((row - 1, col))
                # move right
                if col + 1 < wit.num_cols and wit_state.v_segs[row, col + 1] == 0:
                    neighbors.append((row, col + 1))
                # move left
                if col > 0 and wit_state.v_segs[row, col] == 0:
                    neighbors.append((row, col - 1))
                return neighbors

            neighbors = reachable_neighbors(cell_state)
            for neighbor in neighbors:
                if visited[neighbor] == 1:
                    continue
                this_region.append(neighbor)
                frontier.append(neighbor)
                visited[neighbor] = 1
        regions.append(this_region)
    return regions


if __name__ == "__main__":
    main()
