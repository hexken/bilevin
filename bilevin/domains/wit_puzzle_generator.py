import argparse
from collections import deque
from copy import copy
from itertools import product
import os
from pathlib import Path
import random
import sys

import numpy as np
import tqdm

from args import strtobool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from domains.puzzle_generator import save_problemset
from domains.witness import Witness, WitnessState
from search.loaders import Problem


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--puzzle",
        type=str,
        choices=["triangles", "colors"],
        help="type of witness puzzle to generate",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=lambda p: Path(p).absolute(),
        help="directory path to save problem instances",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=4,
        help="width of puzzles to be generated",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=50000,
        help="number of training puzzles to be generated",
    )
    parser.add_argument(
        "--n-valid",
        type=int,
        default=1000,
        help="number of validation puzzles to be generated",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=1000,
        help="number of testing puzzles to be generated",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "--random-start",
        const=True,
        nargs="?",
        type=strtobool,
        default=False,
        help="use a random border location for the line start, versus the canonical (0, 0) grid location",
    )
    parser.add_argument(
        "--marker-prob",
        type=float,
        default=0.6,
        help="probability of placing a marker in each snake adjacent cell",
    )

    args = parser.parse_args()

    args.output_path.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    border = set()
    for row, col in product((0, args.width), range(args.width + 1)):
        border.add((row, col))
    for row, col in product(range(args.width + 1), (0, args.width)):
        border.add((row, col))
    border = list(border)

    if args.random_start:
        start_choices = border
    else:
        start_choices = [(0, 0)]  # allow fixed start to match PHS paper

    pset_dict_template = {
        "domain_name": "Witness",
        "seed": args.seed,
        "puzzle": args.puzzle,
        "marker_probs": args.marker_prob,
    }

    exclude_problemspecs = set()
    print(f"Saving problems to {args.output_path}")
    if args.n_train > 0:
        with tqdm.tqdm(total=args.n_train) as pbar:
            pbar.set_description("Train problems")
            problems = get_problems(
                args.n_train,
                args.width,
                args.puzzle,
                start_choices,
                border,
                args.marker_prob,
                rng,
                exclude_problemspecs,
                pbar,
            )
        pset_dict = copy(pset_dict_template)
        pset_dict["problems"] = problems
        save_problemset(args.output_path, pset_dict, "train")
        del pset_dict

    if args.n_valid > 0:
        with tqdm.tqdm(total=args.n_valid) as pbar:
            pbar.set_description("Valid problems")
            problems = get_problems(
                args.n_valid,
                args.width,
                args.puzzle,
                start_choices,
                border,
                args.marker_prob,
                rng,
                exclude_problemspecs,
                pbar,
            )
        pset_dict = copy(pset_dict_template)
        pset_dict["problems"] = problems
        save_problemset(args.output_path, pset_dict, "valid")
        del pset_dict

    if args.n_test > 0:
        with tqdm.tqdm(total=args.n_test) as pbar:
            pbar.set_description("Test problems")
            problems = get_problems(
                args.n_test,
                args.width,
                args.puzzle,
                start_choices,
                border,
                args.marker_prob,
                rng,
                exclude_problemspecs,
                pbar,
            )
        pset_dict = copy(pset_dict_template)
        pset_dict["problems"] = problems
        save_problemset(args.output_path, pset_dict, "test")
        del pset_dict


def get_problems(
    n_problems: int,
    width: int,
    puzzle: str,
    start_choices: list,
    goal_choices: list,
    marker_prob: float,
    rng,
    exclude_problemsepcs: set,
    pbar,
):
    problems = []
    id = 0

    while len(problems) < n_problems:
        start_loc = random.choice(start_choices)
        while True:
            goal_loc: tuple = random.choice(goal_choices)
            if goal_loc != start_loc:
                break

        state = WitnessState(
            width=width, head_init_row=start_loc[0], head_init_col=start_loc[1]
        )
        head_at_goal = False
        domain = Witness(
            puzzle=puzzle,
            start_state=state,
            goal_row=goal_loc[0],
            goal_col=goal_loc[1],
            markers=[],
        )
        # generate a path from start to goal
        state = domain.init()
        assert not isinstance(state, list)
        while True:
            avail_actions = domain.actions_unpruned(state)
            if not avail_actions:
                break
            action = rng.choice(avail_actions)
            state = domain.result(state, action)
            if domain.is_head_at_goal(state):
                head_at_goal = True
                break

        if not head_at_goal:
            continue

        # puzzle independent heuristics to make sure the path is not too short
        path_len = state.v_segs.sum() + state.h_segs.sum()
        if path_len < 2.0 * width:
            continue

        if puzzle == "triangles":
            markers = triangle_puzzle_from_path(rng, marker_prob, domain, state)
        elif puzzle == "colors":
            colors = np.arange(1, 5)
            markers = colors_puzzle_from_path(rng, marker_prob, colors, domain, state)
        else:
            raise ValueError(f"Unknown puzzle type {puzzle}")

        if markers is None:
            continue

        hshable = (goal_loc, tuple(markers))
        if hshable in exclude_problemsepcs:
            continue
        else:
            exclude_problemsepcs.add(hshable)

        start_state = WitnessState(width=width, head_init_row=0, head_init_col=0)
        problem_domain = Witness(
            puzzle=puzzle,
            start_state=start_state,
            goal_row=goal_loc[0],
            goal_col=goal_loc[1],
            markers=markers,
        )
        problem = Problem(domain=problem_domain, id=id)
        problems.append(problem)
        id += 1
        pbar.update(1)

    return problems


def connected_components(domain, wit_state):
    """
    Compute the connected components of the grid, i.e. the regions separated by the path
    """
    visited = np.zeros((domain.width, domain.width))
    cell_states = [(i, j) for i, j in product(range(domain.width), range(domain.width))]
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
                if row + 1 < domain.width and wit_state.h_segs[row + 1, col] == 0:
                    neighbors.append((row + 1, col))
                # move down
                if row > 0 and wit_state.h_segs[row, col] == 0:
                    neighbors.append((row - 1, col))
                # move right
                if col + 1 < domain.width and wit_state.v_segs[row, col + 1] == 0:
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


def get_n_adj(state, row, col):
    return int(
        state.v_segs[row, col]
        + state.v_segs[row, col + 1]
        + state.h_segs[row, col]
        + state.h_segs[row + 1, col]
    )


def triangle_puzzle_from_path(
    rng, marker_prob, domain, state
) -> None | list[tuple[int, int, int]]:
    markers = []
    indices = np.array(list(product(range(domain.width), range(domain.width))))
    unused_path_adjacent_indices = []
    for row, col in indices:
        n_adj = get_n_adj(state, row, col)
        if n_adj == 0:
            continue
        elif n_adj == 4:
            domain.plot(state)
            raise ValueError("cell has 4 adjacent segments")
        else:  # n_adj > 0
            if rng.random() <= marker_prob:
                markers.append((row, col, n_adj))
            else:
                unused_path_adjacent_indices.append((row, col))

    # if len(markers) < domain.width // 2:
    #     n = domain.width // 2 - len(markers)

    # make sure at least one marker is placed
    if len(markers) < 1:
        n = 1
        for row, col in rng.choice(unused_path_adjacent_indices, size=n, replace=False):
            n_adj = get_n_adj(state, row, col)
            markers.append((row, col, n_adj))

    return markers


def colors_puzzle_from_path(
    rng, marker_prob, colors, domain, state
) -> None | list[tuple[int, int, int]]:
    regions = connected_components(domain, state)

    if domain.width == 4:
        min_num_regions = 2
    elif domain.width == 5:
        min_num_regions = 3
    elif domain.width == 6:
        min_num_regions = 4
    else:
        raise ValueError(f"Unsupported width {domain.width}")

    # todo heuristic to make sure there are enough regions, maybe avoid repeating actions
    if len(regions) < min_num_regions:
        # print("Not enough regions")
        return None

    n_unit_regions = 0
    for region in regions:
        if len(region) == 1:
            n_unit_regions += 1
        if n_unit_regions > 1:
            return None

    colors = rng.permutation(colors)
    markers = []
    for i, region in enumerate(regions):
        color = colors[i % len(colors)]
        region_indices = np.array(sorted(region))
        marker_mask = rng.random(len(region_indices)) <= marker_prob
        region_markers = region_indices[marker_mask]
        if len(region_markers) > 0:
            markers.extend([(row, col, color) for row, col in region_markers])
        else:
            # make sure at least one marker is placed
            row, col = rng.choice(region_indices, axis=0)
            markers.append((row, col, color))

    return markers


if __name__ == "__main__":
    main()
