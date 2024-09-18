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

from args import strtobool
import domains.pancake as pancake
import domains.stp as stp
from search.loaders import Problem


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--width",
        type=int,
        help="width of puzzles to be generated",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["stp", "pancake"],
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
        "--random-goal",
        const=True,
        nargs="?",
        type=strtobool,
        default=False,
        help="use a random goal state for each problem, instead of the canonical goal state",
    )
    parser.add_argument(
        "--permutation-test",
        const=True,
        nargs="?",
        type=strtobool,
        default=False,
        help="use permutation problems for test set",
    )
    parser.add_argument(
        "--test-steps",
        type=int,
        default=1000,
        help="number of steps from goal for testing puzzles, permutation takes priority",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=50,
        help="min number of steps from goal",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="max number of steps from goal",
    )
    parser.add_argument(
        "--randomize-test-steps",
        const=True,
        nargs="?",
        type=strtobool,
        default=False,
        help="randomize the number of steps away from the goal for the test problems",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=1000,
        help="number of validation puzzles to be generated",
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

    args = parser.parse_args()

    args.output_path.mkdir(parents=True, exist_ok=True)

    exclude_problemspecs = set()

    rng = np.random.default_rng(args.seed)

    pset_dict_template = {"seed": args.seed, "conditional_backward": True}
    pset_dict_template["random_goal"] = args.random_goal
    if args.random_goal:
        pset_dict_template["conditional_forward"] = True
    else:
        pset_dict_template["conditional_forward"] = False

    if args.domain == "stp":
        pset_dict_template["domain"] = "SlidingTile"
        domain_module = stp
        domain_class = stp.SlidingTile
        domain_class = stp.SlidingTile

        for exclude_path in args.exclude_path:
            with Path(exclude_path).open("rb") as f:
                print(f"Excluding problems from {exclude_path}")
                pset = pickle.load(f)
                problems = pset["problems"]
                for problem in problems:
                    exclude_problemspecs.add(problem.domain.start_state)

    elif args.domain == "pancake":
        pset_dict_template["domain"] = "Pancake"
        domain_module = pancake
        domain_class = pancake.Pancake
    else:
        raise ValueError(f"Unknown domain {args.domain}")

    print(f"Saving problems to {args.output_path}")
    if args.n_train > 0:
        with tqdm.tqdm(total=args.n_train) as pbar:
            pbar.set_description("Train problems")
            problems = get_step_problems(
                args.n_train,
                args.width,
                domain_module,
                domain_class,
                args.min_steps,
                args.max_steps,
                True,
                args.random_goal,
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
            problems = get_step_problems(
                args.n_valid,
                args.width,
                domain_module,
                domain_class,
                args.min_steps,
                args.max_steps,
                True,
                args.random_goal,
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
            if args.permutation_test:
                problems = get_permutation_problems(
                    args.n_test,
                    args.width,
                    domain_module,
                    domain_class,
                    rng,
                    exclude_problemspecs,
                    pbar,
                )
            else:
                problems = get_step_problems(
                    args.n_test,
                    args.width,
                    domain_module,
                    domain_class,
                    args.min_steps,
                    args.max_steps,
                    args.randomize_test_steps,
                    args.random_goal,
                    rng,
                    exclude_problemspecs,
                    pbar,
                )
        pset_dict = copy(pset_dict_template)
        pset_dict["problems"] = problems
        if args.permutation_test:
            pset_dict["permutation_test"] = True
        else:
            pset_dict["test_steps"] = args.test_steps
            pset_dict["randomize_test_steps"] = args.randomize_test_steps
        save_problemset(args.output_path, pset_dict, "test")
        del pset_dict


def get_step_problems(
    n_problems: int,
    width: int,
    domain_module,
    domain_class,
    min_steps: int,
    max_steps: int,
    randomize_steps: bool,
    random_goal: bool,
    rng,
    exclude_problemspecs: set,
    pbar,
):

    problems = []
    print(f"  Generating {n_problems} problems with {min_steps}-{max_steps} steps")
    if randomize_steps:
        print("    Randomizing steps")

    id = 0
    while len(problems) < n_problems:
        if randomize_steps:
            steps = rng.integers(min_steps, max_steps + 1)
        else:
            steps = max_steps

        if not random_goal:
            goal_state = domain_module.get_canonical_goal_state(width)
        else:
            goal_state = domain_module.get_permutation(width, rng)
        domain = domain_class(start_state=goal_state)
        state = domain.init()
        for _ in range(steps):
            avail_actions = domain.actions_unpruned(state)
            action = rng.choice(avail_actions)
            state = domain.result(state, action)

        if state == goal_state:
            continue

        if exclude_problemspecs is not None:
            if state in exclude_problemspecs:
                continue
            exclude_problemspecs.add(state)

        new_domain = domain_class(start_state=state, goal_state=goal_state)
        problem = Problem(id=id, domain=new_domain)
        id += 1
        problems.append(problem)
        pbar.update(1)
    return problems


def get_permutation_problems(
    n_problems: int,
    width: int,
    domain_module,
    domain_class,
    rng,
    exclude_problemspecs: set,
    pbar,
):

    problems = []
    print(f"  Generating {n_problems} permutation problems (canonical goal state)")

    id = 0
    while len(problems) < n_problems:
        start_state = domain_module.get_permutation(width, rng)
        if exclude_problemspecs is not None:
            if start_state in exclude_problemspecs:
                continue
            exclude_problemspecs.add(start_state)
        goal_state = domain_module.get_canonical_goal_state(width)
        domain = domain_class(start_state=start_state, goal_state=goal_state)
        problem = Problem(id=id, domain=domain)
        id += 1
        problems.append(problem)
        pbar.update(1)
    return problems


def save_problemset(pth, problemset_dict, suffix):
    n_problems = len(problemset_dict["problems"])
    path = pth / f"{n_problems}-{suffix}.pkl"
    with path.open("wb") as f:
        pickle.dump(problemset_dict, f)
    print(f"  Saved {n_problems} problems to {path}")


if __name__ == "__main__":
    main()
