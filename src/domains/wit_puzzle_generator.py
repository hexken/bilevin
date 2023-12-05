import argparse
from collections import deque
from itertools import product
from pathlib import Path
import random

import numpy as np
import tqdm

from domains.puzzle_generator import save_problemset
from domains.witness import Witness, WitnessState
from search.utils import Problem


def triangle_puzzle_from_path(
    rng, marker_prob, domain, state
) -> None | list[tuple[int, int, int]]:
    markers = []
    for row, col in product(range(domain.width), range(domain.width)):
        n_adj = int(
            state.v_segs[row, col]
            + state.v_segs[row, col + 1]
            + state.h_segs[row, col]
            + state.h_segs[row + 1, col]
        )

        if n_adj == 0:
            continue
        elif n_adj == 4:
            domain.plot(state)
            raise ValueError("cell has 4 adjacent segments")
        if rng.random() <= marker_prob and n_adj >= 1:
            markers.append((row, col, n_adj))

    if len(markers) < domain.width // 2:
        return None

    return markers


def colors_puzzle_from_path(
    rng, marker_prob, domain, state
) -> None | list[tuple[int, int, int]]:
    regions = connected_components(domain, state)

    min_num_regions = 2
    if domain.width == 3:
        min_num_regions = 2
    if domain.width >= 4:
        min_num_regions = 4
    if domain.width == 10:
        min_num_regions = 5

    if len(regions) < min_num_regions:
        return None

    # fill regions with colors, only keep sufficiently non empty ones
    colors = random.choices(range(1, domain.max_num_colors + 1), k=len(regions))
    unique_colors_used = set()
    markers = []
    non_unit_regions_unique_colors = 0
    for region in regions:
        region_arr = np.array(sorted(region))
        region_mask = rng.random(len(region_arr)) < marker_prob
        region_arr = region_arr[region_mask]
        if len(region_arr):
            color = colors.pop()
            if len(region_arr) > 1 and color not in unique_colors_used:
                non_unit_regions_unique_colors += 1
            unique_colors_used.add(color)
            markers.extend([(row, col, color) for row, col in region_arr])

    if non_unit_regions_unique_colors < domain.width // 2:
        return None
    return markers


def generate_problems(
    args,
    rng,
    goal_choices,
    n_problems: int,
    id_counter_start: int,
    exclude_problemsepcs: set,
    pbar,
):
    problems = []
    id = id_counter_start
    init_state = WitnessState(width=args.width, head_init_row=0, head_init_col=0)

    while len(problems) < n_problems:
        head_at_goal = False
        goal = random.choice(goal_choices)
        domain = Witness(
            puzzle=args.puzzle,
            initial_state=init_state,
            goal_row=goal[0],
            goal_col=goal[1],
            markers=[],
        )
        # generate a path from start to goal
        state: WitnessState = domain.reset()
        while True:
            actions, _ = domain.actions_unpruned(state)
            if not actions:
                break
            action = rng.choice(actions)
            state = domain.result(state, action)
            if domain.is_head_at_goal(state):
                head_at_goal = True
                break

        if not head_at_goal:
            continue

        # heuristic to make sure the path is not too short
        if (state.v_segs.sum() + state.h_segs.sum()) / state.grid.shape[
            0
        ] ** 2 < args.min_path_ratio:
            continue

        if args.puzzle == "triangles":
            markers = triangle_puzzle_from_path(rng, args.marker_prob, domain, state)
        elif args.puzzle == "colors":
            markers = colors_puzzle_from_path(rng, args.marker_prob, domain, state)
        else:
            raise ValueError(f"Unknown puzzle type {args.puzzle}")

        if markers is None:
            continue

        hshable = (goal, tuple(markers))
        if hshable in exclude_problemsepcs:
            continue
        else:
            exclude_problemsepcs.add(hshable)

        problem_init_state = WitnessState(
            width=args.width, head_init_row=0, head_init_col=0
        )
        problem_domain = Witness(
            puzzle=args.puzzle,
            initial_state=problem_init_state,
            goal_row=goal[0],
            goal_col=goal[1],
            markers=markers,
        )
        problem = Problem(domain=problem_domain, id=id)
        problems.append(problem)
        id += 1
        pbar.update(1)

    return problems


def main():
    """
    Generate a dataset of Witness colors problems. A generated problem instance is only kept if
    there least args.width // 2 colored regions of size at least 2, each with a unique color.
    """
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

    # parser.add_argument(
    #     "-c",
    #     "--max-num-colors",
    #     type=int,
    #     default=4,
    #     help="number of colors to use",
    # )
    parser.add_argument(
        "-p",
        "--marker-prob",
        type=float,
        default=0.6,
        help="probability of placing a marker in each snake adjacent cell",
    )
    parser.add_argument(
        "--min-path-ratio",
        type=float,
        default=0.8,
        help="path that generated problem must be >= this ratio of squared width",
    )

    args = parser.parse_args()

    args.output_path.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    goals_dups = (
        [(0, i) for i in range(args.width + 1)]
        + [(i, 0) for i in range(args.width + 1)]
        + [(args.width, i) for i in range(args.width + 1)]
        + [(i, args.width) for i in range(args.width + 1)]
    )
    goals = set()
    for goal in goals_dups:
        goals.add(goal)
    goals.remove((0, 0))
    goals = list(goals)

    problem_specs = set()

    dummy_init_state = WitnessState(width=args.width, head_init_row=0, head_init_col=0)
    dummy_domain = Witness(
        puzzle=args.puzzle,
        initial_state=dummy_init_state,
        goal_row=0,
        goal_col=0,
        markers=[],
    )
    dummy_domain.reset()
    problemset_dict = {
        "domain_name": "Witness",
        "state_t_width": dummy_domain.state_width,
        "in_channels": dummy_domain.in_channels,
        "num_actions": dummy_domain.num_actions,
        "seed": args.seed,
        "puzzle": args.puzzle,
        "marker_prob": args.marker_prob,
        "min_path_ratio": args.min_path_ratio,
    }

    with tqdm.tqdm(total=args.n_train) as pbar:
        pbar.set_description("Curriculum problems")
        curriculum_problems = generate_problems(
            args,
            rng,
            goals,
            args.n_train,
            0,
            problem_specs,
            pbar,
        )
    problemset_dict["problems"] = [curriculum_problems]
    save_problemset(args.output_path, problemset_dict, "train")

    if args.n_valid > 0:
        with tqdm.tqdm(total=args.n_valid) as pbar:
            pbar.set_description("Valid problems")
            valid_problems = generate_problems(
                args,
                rng,
                goals,
                args.n_valid,
                0,
                problem_specs,
                pbar,
            )
            problemset_dict["problems"] = [valid_problems]
        save_problemset(
            args.output_path,
            problemset_dict,
            "valid",
        )

    if args.n_test > 0:
        with tqdm.tqdm(total=args.n_test) as pbar:
            pbar.set_description("Test problems")
            test_problems = generate_problems(
                args,
                rng,
                goals,
                args.n_test,
                0,
                problem_specs,
                pbar,
            )
            problemset_dict["problems"] = [test_problems]
        save_problemset(
            args.output_path,
            problemset_dict,
            "test",
        )


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


if __name__ == "__main__":
    main()
