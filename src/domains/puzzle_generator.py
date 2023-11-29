import argparse
from copy import copy
from pathlib import Path
import pickle

import numpy as np
import tqdm

from domains.cube3 import Cube3
from domains.cube3 import get_goal_state as cube3ggs
from domains.stp import SlidingTilePuzzle
from domains.stp import get_goal_state as stpggs
from search.utils import Problem


def save_problemset(pth, problemset_dict, suffix):
    n_problems = sum(len(p) for p in problemset_dict["problems"])
    path = pth / f"{n_problems}-{suffix}.pkl"
    with path.open("wb") as f:
        pickle.dump(problemset_dict, f)
    print(f"Saved {n_problems} problems to {path}")


def generate_step_problems(
    init_domain,
    domain_class,
    rng,
    n_problems: int,
    min_steps: int,
    max_steps: int,
    id_counter_start: int,
    exclude_problemspecs: set,
    randomize: bool,
    pbar,
):
    problems = []
    problems_generated = 0
    id_counter = id_counter_start
    while problems_generated < n_problems:
        if randomize:
            steps = rng.integers(min_steps, max_steps + 1)
        else:
            steps = max_steps
        state = init_domain.reset()
        for _ in range(steps):
            actions, _ = init_domain.actions_unpruned(state)
            random_action = rng.choice(actions)
            state = init_domain.result(state, random_action)

        if state in exclude_problemspecs or init_domain.is_goal(state):
            continue
        else:
            exclude_problemspecs.add(state)
            new_domain = domain_class(initial_state=state)
            problem = Problem(id=id_counter, domain=new_domain)
            problems.append(problem)
            problems_generated += 1
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
        choices=["cube3", "stp"],
        help="domain",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=lambda p: Path(p).absolute(),
        help="path to save problem instances",
    )
    parser.add_argument(
        "--randomize-curriculum-steps",
        action="store_true",
        default=False,
        help="randomize the number of steps away from the goal for the curriculum",
    )
    parser.add_argument(
        "--final-stage",
        action="store_true",
        default=False,
        help="insert a final stage with the maximum number of steps away from the goal, not randomized",
    )
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
        "--num-stages",
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

    if args.domain == "cube3":
        problemset_dict = {
            "domain_name": "Cube3",
            "state_t_width": 3,
            "state_t_depth": 6,
            "num_actions": 12,
            "in_channels": 6,
            "requires_backward_goal": True,
            "seed": args.seed,
        }
        domain = Cube3(cube3ggs())
        domain_class = Cube3
    elif args.domain == "stp":
        problemset_dict = {
            "domain_name": "SlidingTilePuzzle",
            "state_t_width": args.width,
            "state_t_depth": 1,
            "num_actions": 4,
            "in_channels": int(args.width**2),
            "requires_backward_goal": True,
            "seed": args.seed,
        }
        domain = SlidingTilePuzzle(stpggs(args.width))
        domain_class = SlidingTilePuzzle
    else:
        raise ValueError(f"Unknown domain {args.domain}")

    if args.stages_multiple > 0 and args.num_stages > 0:
        stages = [args.stages_multiple * i for i in range(1, args.num_stages + 1)]
    elif len(args.stages) > 0:
        stages = args.stages
    else:
        raise ValueError("Must specify either stages or stages_multiple and num_stages")

    print(
        f"Generating {args.n_problems_per_stage} problems for each of {len(stages)} stages: {stages}"
    )

    curriculum_problems = []
    total_num_curriculum_problems = len(stages) * args.n_problems_per_stage + (
        args.n_problems_per_stage if args.final_stage else 0
    )
    num_curriculum_problems = 0
    with tqdm.tqdm(total=total_num_curriculum_problems) as pbar:
        pbar.set_description("Curriculum problems")
        for i in range(len(stages)):
            minsteps = stages[i - 1] + 1 if i > 0 else 1
            maxsteps = stages[i]
            stage_problems = generate_step_problems(
                domain,
                domain_class,
                rng,
                args.n_problems_per_stage,
                minsteps,
                maxsteps,
                num_curriculum_problems,
                exclude_problemspecs,
                args.randomize_curriculum_steps,
                pbar,
            )
            curriculum_problems.append(stage_problems)
            num_curriculum_problems += len(stage_problems)
        if args.final_stage:
            stage_problems = generate_step_problems(
                domain,
                domain_class,
                rng,
                args.n_problems_per_stage,
                stages[-1],
                stages[-1],
                num_curriculum_problems,
                exclude_problemspecs,
                False,
                pbar,
            )
            curriculum_problems.append(stage_problems)
            num_curriculum_problems += len(stage_problems)

    trainset_dict = copy(problemset_dict)
    trainset_dict["randomize_steps"] = args.randomize_curriculum_steps
    trainset_dict["stages"] = stages
    trainset_dict["problems_per_stage"] = args.n_problems_per_stage
    trainset_dict["problems"] = curriculum_problems
    save_problemset(args.output_path, trainset_dict, "train")

    if args.n_valid > 0:
        with tqdm.tqdm(total=args.n_valid) as pbar:
            pbar.set_description("Valid problems")
            valid_problems = generate_step_problems(
                domain,
                domain_class,
                rng,
                args.n_valid,
                args.test_steps,
                args.test_steps,
                0,
                exclude_problemspecs,
                args.randomize_test_steps,
                pbar,
            )
            problemset_dict["problems"] = [valid_problems]
            problemset_dict["randomize_steps"] = args.randomize_test_steps
            problemset_dict["test_steps"] = args.test_steps
        save_problemset(
            args.output_path,
            problemset_dict,
            "valid",
        )

    if args.n_test > 0:
        with tqdm.tqdm(total=args.n_test) as pbar:
            pbar.set_description("Test problems")
            test_problems = generate_step_problems(
                domain,
                domain_class,
                rng,
                args.n_test,
                args.test_steps,
                args.test_steps,
                0,
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
