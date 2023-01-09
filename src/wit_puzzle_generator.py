import argparse
from collections import deque
from itertools import product
import json
from pathlib import Path
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
        "--max-num-colors",
        type=int,
        default=4,
        help="number of colors to use",
    )

    parser.add_argument(
        "-p",
        "--bullet-prob",
        type=float,
        default=0.6,
        help="probability of placing a bullet in each empty cell",
    )

    args = parser.parse_args()

    assert args.max_num_colors >= 2, "Number of colors must be at least 2"

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

    problem_specs = set()

    dataset = {
        "domain_module": "witness",
        "domain_name": "Witness",
        "width": args.width,
        "max_num_colors": args.max_num_colors,
        "problems": [],
    }

    problem_id = 0
    with tqdm(total=args.num_problems) as pbar:
        while len(problem_specs) < args.num_problems:
            head_at_goal = False
            goal = random.choice(goals)
            problem = {
                "init": (0, 0),  # todo allow other initial pos?
                "goal": goal,
            }
            wit = Witness(
                width=args.width, max_num_colors=args.max_num_colors, **problem
            )
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
            if len(regions) < min_num_regions or len(regions) > args.max_num_colors:
                continue

            # fill each region with a color
            colors = random.sample(range(1, args.max_num_colors + 1), len(regions))
            colored_cells = []
            for region in regions:
                region_arr = np.array(sorted(region))
                region_mask = np.random.rand(len(region_arr)) < args.bullet_prob
                region_arr = region_arr[region_mask]
                if len(region_arr):
                    colored_cells.extend(
                        [f"{row} {col} {colors[-1]}" for row, col in region_arr]
                    )
                    colors.pop()

            # Only add if at least two colors were used
            max_num_colors_used = len(regions) - len(colors)
            if max_num_colors_used < 2:
                continue

            problem["colored_cells"] = colored_cells
            problem_str = str(problem)
            if problem_str in problem_specs:
                continue
            else:
                problem_specs.add(problem_str)
                problem["id"] = problem_id
                problem_id += 1
                dataset["problems"].append(problem)
                pbar.update()

    with args.output_path.open("w") as f:
        json.dump(dataset, f, indent=2)


def connected_components(wit, wit_state):
    """
    Compute the connected components of the grid, i.e. the regions separated by the path
    """
    visited = np.zeros((wit.width, wit.width))
    cell_states = [(i, j) for i, j in product(range(wit.width), range(wit.width))]
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
                if row + 1 < wit.width and wit_state.h_segs[row + 1, col] == 0:
                    neighbors.append((row + 1, col))
                # move down
                if row > 0 and wit_state.h_segs[row, col] == 0:
                    neighbors.append((row - 1, col))
                # move right
                if col + 1 < wit.width and wit_state.v_segs[row, col + 1] == 0:
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
