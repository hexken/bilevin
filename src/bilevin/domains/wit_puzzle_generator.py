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

# todo allow randomized start location?


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

    # make sure at least width // 2 markers present
    if len(markers) < domain.width // 2:
        n = domain.width // 2 - len(markers)
        for row, col in rng.choice(unused_path_adjacent_indices, size=n, replace=False):
            n_adj = get_n_adj(state, row, col)
            markers.append((row, col, n_adj))

    return markers


def colors_puzzle_from_path(
    rng, marker_prob, colors, domain, state
) -> None | list[tuple[int, int, int]]:
    regions = connected_components(domain, state)

    if domain.width == 4:
        min_num_regions = 3
    elif domain.width == 5:
        min_num_regions = 4
    elif domain.width == 6:
        min_num_regions = 5

    if len(regions) < min_num_regions:
        # todo heuristic to make sure there are enough regions, maybe avoid repeating actions
        # print("Not enough regions")
        return None

    # fill regions with colors, only keep sufficiently non empty ones
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


def generate_problems(
    args,
    rng,
    goal_choices,
    n_problems: int,
    marker_prob: float,
    min_path_ratio: float,
    id_counter_start: int,
    exclude_problemsepcs: set,
    pbar,
):
    problems = []
    id = id_counter_start
    init_state = WitnessState(width=args.width, head_init_row=0, head_init_col=0)
    colors = np.arange(1, 5)

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
        actions, _ = domain.actions_unpruned(state)
        action = rng.choice(actions)
        state = domain.result(state, action)
        while True:
            actions, _ = domain.actions(action, state)
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
        if (state.v_segs.sum() + state.h_segs.sum()) / (
            state.head_row + state.head_col
        ) < min_path_ratio:
            # print("Path too short")
            continue

        if args.puzzle == "triangles":
            markers = triangle_puzzle_from_path(rng, marker_prob, domain, state)
        elif args.puzzle == "colors":
            markers = colors_puzzle_from_path(rng, marker_prob, colors, domain, state)
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
        "--n-problems-per-stage",
        type=int,
        default=50000,
        help="number of training puzzles to be generated",
    )
    parser.add_argument(
        "--marker-prob-limits",
        type=float,
        nargs="+",
        default=[0.3, 0.6],
        help="probability of placing a marker in each snake adjacent cell",
    )
    parser.add_argument(
        "--test-marker-prob",
        type=float,
        default=0.6,
        help="probability of placing a marker in each snake adjacent cell",
    )
    parser.add_argument(
        "--min-path-ratio-limits",
        type=float,
        nargs="+",
        default=[1.0, 1.5],
        help="path that generated problem must be >= this ratio of squared width",
    )
    parser.add_argument(
        "--test-min-path-ratio",
        type=float,
        default=1.0,
        help="path that generated problem must be >= this ratio of squared width",
    )
    parser.add_argument(
        "--n-stages",
        type=int,
        default=9,
        help="number of stages in the curriculum",
    )
    parser.add_argument(
        "--n-problems-final-stage",
        type=int,
        default=-1,
        help="number of problems for the final stage. Set to -1 to use n-problems-per-stage. Final stage problems are generated in the same way as the test problems.",
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

    curriculum_problems = []
    num_curriculum_problems = 0
    assert len(args.marker_prob_limits) == 2
    assert len(args.min_path_ratio_limits) == 2

    marker_probs = np.linspace(
        args.marker_prob_limits[0], args.marker_prob_limits[1], args.n_stages - 1
    )
    min_path_ratios = np.linspace(
        args.min_path_ratio_limits[0], args.min_path_ratio_limits[1], args.n_stages - 1
    )

    trainset_dict = {
        "domain_name": "Witness",
        "seed": args.seed,
        "puzzle": args.puzzle,
        "marker_probs": marker_probs.tolist(),
        "min_path_ratios": min_path_ratios.tolist(),
        "n_problems_per_stage": args.n_problems_per_stage,
        "n_problems_final_stage": args.n_problems_final_stage,
    }

    n_problems_final_stage = (
        args.n_problems_per_stage
        if args.n_problems_final_stage < 0
        else args.n_problems_final_stage
    )
    total_num_curriculum_problems = (
        len(marker_probs) * args.n_problems_per_stage + n_problems_final_stage
    )

    print(f"Saving problems to {args.output_path}")
    print(f"  {args.n_problems_per_stage} problems for each of {len(marker_probs)}")
    print(f" marker probs: {marker_probs.tolist()}")
    print(f" min path ratios: {min_path_ratios.tolist()}")
    print(f"  {n_problems_final_stage} problems for final stage.")

    with tqdm.tqdm(total=total_num_curriculum_problems) as pbar:
        pbar.set_description("Curriculum problems")
        for i in range(len(marker_probs)):
            stage_problems = generate_problems(
                args,
                rng,
                goals,
                args.n_problems_per_stage,
                marker_probs[i],
                min_path_ratios[i],
                num_curriculum_problems,
                problem_specs,
                pbar,
            )
            curriculum_problems.append(stage_problems)
            num_curriculum_problems += len(stage_problems)
        if n_problems_final_stage > 0:
            stage_problems = generate_problems(
                args,
                rng,
                goals,
                n_problems_final_stage,
                args.test_marker_prob,
                args.test_min_path_ratio,
                num_curriculum_problems,
                problem_specs,
                pbar,
            )
            curriculum_problems.append(stage_problems)
            num_curriculum_problems += len(stage_problems)
    trainset_dict["problems"] = curriculum_problems
    save_problemset(args.output_path, trainset_dict, "train")

    testset_dict = {
        "domain_name": "Witness",
        "seed": args.seed,
        "puzzle": args.puzzle,
        "marker_prob": args.test_marker_prob,
        "min_path_ratio": args.test_min_path_ratio,
    }
    if args.n_valid > 0:
        with tqdm.tqdm(total=args.n_valid) as pbar:
            pbar.set_description("Valid problems")
            valid_problems = generate_problems(
                args,
                rng,
                goals,
                args.n_valid,
                args.test_marker_prob,
                args.test_min_path_ratio,
                0,
                problem_specs,
                pbar,
            )
            testset_dict["problems"] = [valid_problems]
        save_problemset(
            args.output_path,
            testset_dict,
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
                args.test_marker_prob,
                args.test_min_path_ratio,
                0,
                problem_specs,
                pbar,
            )
            testset_dict["problems"] = [test_problems]
        save_problemset(
            args.output_path,
            testset_dict,
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
