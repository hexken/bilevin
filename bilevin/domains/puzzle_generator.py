import argparse
from copy import copy
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from domains.pancake import Pancake, PancakeState
from domains.pancake import get_goal_state as pancakeggs
from domains.stp import SlidingTile, SlidingTileState
from domains.stp import get_goal_state as stpggs
from search.loaders import Problem

# from domains.cube3 import Cube3
# from domains.cube3 import get_goal_state as cube3ggs
print(sys.path)


def save_problemset(pth, problemset_dict, suffix):
    n_problems = sum(len(p) for p in problemset_dict["problems"])
    path = pth / f"{n_problems}-{suffix}.pkl"
    with path.open("wb") as f:
        pickle.dump(problemset_dict, f)
    print(f"  Saved {n_problems} problems to {path}")


def is_solvable(tiles):
    n = tiles.shape[0]
    flat_tiles = tiles.flatten()
    flat_tiles = flat_tiles[flat_tiles != 0]
    inversions = 0
    for i in range(len(flat_tiles)):
        for j in range(i + 1, len(flat_tiles)):
            if flat_tiles[i] > flat_tiles[j]:
                inversions += 1
    if n % 2 == 1:
        return inversions % 2 == 0
    else:
        blank_row_from_top = np.where(tiles == 0)[0][0] + 1
        return (blank_row_from_top % 2 == 0) != (inversions % 2 == 0)


def random_pancake_puzzles(
    rng, width, n_problems, id_counter_start, exclude_problemspecs, pbar
):
    problems = []
    id_counter = id_counter_start
    while len(problems) < n_problems:
        pancakes = rng.permutation(width)
        state = PancakeState(pancakes)
        if exclude_problemspecs is not None:
            if state in exclude_problemspecs:
                continue
            exclude_problemspecs.add(state)

        new_domain = Pancake(initial_state=state)
        problem = Problem(id=id_counter, domain=new_domain)
        problems.append(problem)
        id_counter += 1
        pbar.update(1)
    return problems


def random_sliding_tile_puzzles(
    rng, width, n_problems, id_counter_start, exclude_problemspecs, pbar
):
    problems = []
    id_counter = id_counter_start
    while len(problems) < n_problems:
        tiles = rng.permutation(width**2).reshape(width, width)
        if not is_solvable(tiles):
            continue
        r, c = np.where(tiles == 0)
        r = r.item()
        c = c.item()
        state = SlidingTileState(tiles, r, c)
        if exclude_problemspecs is not None:
            if state in exclude_problemspecs:
                continue
            exclude_problemspecs.add(state)

        new_domain = SlidingTile(start_state=state)
        problem = Problem(id=id_counter, domain=new_domain)
        problems.append(problem)
        id_counter += 1
        pbar.update(1)
    return problems


def generate_step_problems(
    init_domain,
    domain_class,
    rng,
    n_problems: int,
    min_steps: int,
    max_steps: int,
    id_counter_start: int,
    check_exclude: bool,
    exclude_problemspecs: set,
    randomize: bool,
    pbar,
):

    problems = []
    id_counter = id_counter_start
    print(f"  Generating {n_problems} problems with {min_steps}-{max_steps} steps")
    while len(problems) < n_problems:
        if randomize:
            steps = rng.integers(min_steps, max_steps + 1)
        else:
            steps = max_steps
        state = init_domain.init()
        for _ in range(steps):
            avail_actions = init_domain.actions_unpruned(state)
            action = rng.choice(avail_actions)
            state = init_domain.result(state, action)

        if init_domain.is_goal(state):
            continue

        if exclude_problemspecs is not None:
            if check_exclude and state in exclude_problemspecs:
                continue
            exclude_problemspecs.add(state)

        new_domain = domain_class(start_state=state)
        problem = Problem(id=id_counter, domain=new_domain)
        problems.append(problem)
        id_counter += 1
        pbar.update(1)
    return problems


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--width",
        type=int,
        default=4,
        help="width of puzzles to be generated",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["cube3", "stp", "pancake"],
        help="domain",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=lambda p: Path(p).absolute(),
        help="path to save problem instances",
    )
    parser.add_argument(
        "--exclude-path",
        default=[],
        action="append",
        help="path of problems to exclude for stp",
    )
    parser.add_argument(
        "--test-permutation",
        action="store_true",
        default=False,
        help="use permutation problems for stp valid",
    )
    parser.add_argument(
        "--randomize-curriculum-steps",
        action="store_true",
        default=False,
        help="randomize the number of steps away from the goal for the curriculum",
    )
    parser.add_argument(
        "--increasing-minstep",
        action="store_true",
        default=False,
        help="lower bound on steps away from goal increases with each stage",
    )
    # parser.add_argument(
    #     "--final-stage",
    #     action="store_true",
    #     default=False,
    #     help="insert a final stage with the maximum number of steps away from the goal, not randomized",
    # )
    parser.add_argument(
        "--n-problems-final-stage",
        type=int,
        default=-1,
        help="number of problems for the final stage. Set to -1 to use n-problems-per-stage. Final stage problems are generated in the same way as the test problems.",
    )
    # parser.add_argument(
    #     "--randomize-final-stage",
    #     action="store_true",
    #     default=False,
    #     help="randomize the number of steps away from the goal for the final stage",
    # )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=[],
        help="list of steps away from goal for the curriculum",
    )
    parser.add_argument(
        "--stages-multiple",
        type=int,
        default=1000,
        help="multiple of steps to use when generating a curriculum",
    )
    parser.add_argument(
        "--n-stages",
        type=int,
        default=1,
        help="number of stages to use when generating a curriculum",
    )
    parser.add_argument(
        "--n-problems-per-stage",
        type=int,
        default=50000,
        help="number of training puzzles to be generated",
    )
    parser.add_argument(
        "--test-steps",
        type=int,
        default=1000,
        help="number of steps from goal for testing puzzles",
    )
    parser.add_argument(
        "--randomize-test-steps",
        action="store_true",
        default=False,
        help="randomize the number of steps away from the goal for the test problems",
    )
    parser.add_argument(
        "--n-valid",
        type=int,
        default=4000,
        help="number of validation puzzles to be generated",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=4000,
        help="number of testing puzzles to be generated",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )

    args = parser.parse_args()

    args.output_path.mkdir(parents=True, exist_ok=True)

    exclude_problemspecs = set()

    rng = np.random.default_rng(args.seed)

    problemset_dict = {"seed": args.seed}
    # if args.domain == "cube3":
    #     problemset_dict["domain_name"] = "Cube3"
    #     domain = Cube3(cube3ggs())
    if args.domain == "stp":
        random_puzzle_function = random_sliding_tile_puzzles
        problemset_dict["domain_name"] = "SlidingTile"
        domain = SlidingTile(stpggs(args.width))
        for exclude_path in args.exclude_path:
            with Path(exclude_path).open("rb") as f:
                print(f"Excluding problems from {exclude_path}")
                pset = pickle.load(f)
                problems = pset["problems"][0]
                for problem in problems:
                    exclude_problemspecs.add(problem.domain.start_state)
    elif args.domain == "pancake":
        random_puzzle_function = random_pancake_puzzles
        problemset_dict["domain_name"] = "Pancake"
        domain = Pancake(pancakeggs(args.width))
    else:
        raise ValueError(f"Unknown domain {args.domain}")
    assert domain.is_goal(domain.init())
    domain_class = domain.__class__

    if args.stages_multiple > 0 and args.n_stages > 0:
        stages = [args.stages_multiple * i for i in range(1, args.n_stages)]
        n_stages = args.n_stages
    elif len(args.stages) > 0:
        stages = [int(s) for s in args.stages]
        n_stages = len(stages) - 1
    else:
        stages = []
        n_stages = 0

    print(f"Increasing minstep: {args.increasing_minstep}")
    print(f"Saving problems to {args.output_path}")
    print(
        f"  {args.n_problems_per_stage} problems for each of {n_stages} stages: {stages}"
    )

    curriculum_problems = []
    n_problems_final_stage = (
        args.n_problems_per_stage
        if args.n_problems_final_stage < 0
        else args.n_problems_final_stage
    )
    print(f"  {n_problems_final_stage} problems for final stage.")
    total_num_curriculum_problems = (
        n_stages * args.n_problems_per_stage + n_problems_final_stage
    )
    num_curriculum_problems = 0
    if total_num_curriculum_problems > 0:
        with tqdm.tqdm(total=total_num_curriculum_problems) as pbar:
            pbar.set_description("Curriculum problems")

            for i in range(n_stages):
                if args.stages:
                    minsteps = stages[i]
                    maxsteps = stages[i + 1]
                else:
                    minsteps = (
                        stages[i - 1] + 1 if (i > 0 and args.increasing_minstep) else 1
                    )
                    maxsteps = stages[i]

                if args.domain == "cube3" and maxsteps <= 10:
                    check_exclude = False
                elif args.domain == "pancake" and maxsteps < args.width:
                    check_exclude = False
                elif args.domain == "stp" and maxsteps <= 10:
                    check_exclude = False
                else:
                    check_exclude = True
                stage_problems = generate_step_problems(
                    domain,
                    domain_class,
                    rng,
                    args.n_problems_per_stage,
                    minsteps,
                    maxsteps,
                    num_curriculum_problems,
                    check_exclude,
                    exclude_problemspecs,
                    args.randomize_curriculum_steps,
                    pbar,
                )
                curriculum_problems.append(stage_problems)
                num_curriculum_problems += len(stage_problems)
            if n_problems_final_stage > 0:
                stage_problems = generate_step_problems(
                    domain,
                    domain_class,
                    rng,
                    n_problems_final_stage,
                    1,
                    args.test_steps,
                    num_curriculum_problems,
                    True,
                    exclude_problemspecs,
                    args.randomize_test_steps,
                    pbar,
                )
                curriculum_problems.append(stage_problems)
                num_curriculum_problems += len(stage_problems)

        trainset_dict = copy(problemset_dict)
        trainset_dict["randomize_steps"] = args.randomize_curriculum_steps
        trainset_dict["stages"] = stages
        trainset_dict["problems_per_stage"] = args.n_problems_per_stage
        trainset_dict["n_problems_final_stage"] = n_problems_final_stage
        trainset_dict["problems"] = curriculum_problems
        save_problemset(args.output_path, trainset_dict, "train")

    if args.n_valid > 0:
        with tqdm.tqdm(total=args.n_valid) as pbar:
            pbar.set_description("Valid problems")
            if args.test_permutation:
                valid_problems = random_puzzle_function(
                    rng,
                    args.width,
                    args.n_valid,
                    0,
                    exclude_problemspecs,
                    pbar,
                )
            else:
                valid_problems = generate_step_problems(
                    domain,
                    domain_class,
                    rng,
                    args.n_valid,
                    1,
                    args.test_steps,
                    0,
                    True,
                    exclude_problemspecs,
                    args.randomize_test_steps,
                    pbar,
                )
                problemset_dict["randomize_steps"] = args.randomize_test_steps
                problemset_dict["test_steps"] = args.test_steps
            problemset_dict["problems"] = [valid_problems]
        save_problemset(
            args.output_path,
            problemset_dict,
            "valid",
        )

    if args.n_test > 0:
        with tqdm.tqdm(total=args.n_test) as pbar:
            pbar.set_description("Test problems")
            if args.test_permutation:
                test_problems = random_puzzle_function(
                    rng,
                    args.width,
                    args.n_test,
                    0,
                    exclude_problemspecs,
                    pbar,
                )
            else:
                test_problems = generate_step_problems(
                    domain,
                    domain_class,
                    rng,
                    args.n_test,
                    1,
                    args.test_steps,
                    0,
                    True,
                    exclude_problemspecs,
                    args.randomize_test_steps,
                    pbar,
                )
            problemset_dict["problems"] = [test_problems]
            problemset_dict["randomize_steps"] = args.randomize_test_steps
            problemset_dict["test_steps"] = args.test_steps
        save_problemset(
            args.output_path,
            problemset_dict,
            "test",
        )


if __name__ == "__main__":
    main()
