# Copyright (C) 2021-2022, Ken Tjhia
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import argparse
from copy import copy
import json
from pathlib import Path

import numpy as np
import tqdm

from domains.cube3 import Cube3, Cube3State, get_goal_state, goal_check


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--output-path",
        type=lambda p: Path(p).absolute(),
        help="path to save problem instances",
    )
    parser.add_argument(
        "--bootstrap-steps",
        type=int,
        default=10,
        help="generate all problems up to (inclusive) this many steps away from the goal",
    )
    parser.add_argument(
        "--pad-bootstrap-problems",
        type=int,
        default=40,
        help="pad bootstrap problems by resampling to be the smallest multiple of this, 0 for no pad",
    )

    parser.add_argument(
        "--randomize-curriculum-steps",
        action="store_true",
        default=False,
        help="randomize the number of steps away from the goal for the curriculum",
    )
    parser.add_argument(
        "--curriculum",
        nargs="+",
        default=[],
        help="list of steps away from goal for the curriculum",
    )
    parser.add_argument(
        "--curriculum-multiple",
        type=int,
        default=0,
        help="multiple of steps to use when generating a curriculum",
    )
    parser.add_argument(
        "--curriculum-num-difficulties",
        type=int,
        default=0,
        help="number of difficulties to use when generating a curriculum",
    )
    parser.add_argument(
        "--n-problems-per-difficulty",
        type=int,
        default=3200,
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
        default=3200,
        help="number of validation puzzles to be generated",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=3200,
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
    goal_state = get_goal_state()
    cube3 = Cube3(goal_state)

    problemset_dict = {
        "domain_module": "cube3",
        "domain_name": "Cube3",
        "state_t_width": 3,
        "num_actions": 12,
        "in_channels": 6,
        "seed": args.seed,
    }

    def get_all_reachable_states(
        id_prefix, id_start, state, max_step, exclude_problemspecs
    ):
        problems = []
        id_counter = id_start

        def helper(state, step, max_step):
            nonlocal id_counter
            if step > max_step:
                return
            cube3 = Cube3(goal_state)
            actions, _ = cube3.actions_unpruned(state)
            for action in actions:
                new_state = cube3.result(state, action)
                if new_state in exclude_problemspecs or cube3.is_goal(new_state):
                    continue
                else:
                    exclude_problemspecs.add(new_state)
                    problem = {
                        "front": new_state.front.tolist(),
                        "up": new_state.up.tolist(),
                        "down": new_state.down.tolist(),
                        "left": new_state.left.tolist(),
                        "right": new_state.right.tolist(),
                        "back": new_state.back.tolist(),
                        "id": f"{id_prefix}_{id_counter}",
                    }
                    id_counter += 1
                    problems.append(problem)
                    helper(new_state, step + 1, max_step)

        helper(state, 1, max_step)
        return problems

    def save_problemset(problemset_dict, suffix):
        n_problems = 0
        if "problems" in problemset_dict:
            n_problems += len(problemset_dict["problems"])
        if "bootstrap_problems" in problemset_dict:
            n_problems += len(problemset_dict["bootstrap_problems"])
        if "curriculum_problems" in problemset_dict:
            n_problems += len(problemset_dict["curriculum_problems"])

        path = args.output_path / f"{n_problems}-{suffix}.json"
        with path.open("w") as f:
            json.dump(problemset_dict, f)
        print(f"Saved {n_problems} problems to {path}")

    print(
        f"Generating bootstrap problems up to {args.bootstrap_steps} steps away from goal.."
    )
    bootstrap_problems = get_all_reachable_states(
        "b", 0, cube3.initial_state, args.bootstrap_steps, exclude_problemspecs
    )
    print(f"Generated {len(bootstrap_problems)} problems.")
    if args.pad_bootstrap_problems > 0:
        nearest_multiple = (
            (len(bootstrap_problems) // args.pad_bootstrap_problems) + 1
        ) * args.pad_bootstrap_problems
        k_add = nearest_multiple - len(bootstrap_problems)
        print(
            f"Padding bootstrap problems with {k_add} for a total of {nearest_multiple} problems."
        )
        indices = rng.choice(np.arange(len(bootstrap_problems)), size=k_add)
        for i in indices:
            bootstrap_problems.append(bootstrap_problems[i])

    if args.curriculum_multiple > 0 and args.curriculum_num_difficulties > 0:
        curriculum = [
            args.curriculum_multiple * i
            for i in range(1, args.curriculum_num_difficulties + 1)
        ]
    elif len(args.curriculum) > 0:
        curriculum = args.curriculum
    else:
        raise ValueError(
            "Must specify either curriculum or curriculum_multiple and curriculum_num_difficulties or a curriculum"
        )

    print(
        f"Generating {args.n_problems_per_difficulty} curriculum problems for each of {len(curriculum)} steps: {curriculum}"
    )

    def generate_step_problems(
        n_problems: int,
        max_steps: int,
        id_prefix: str,
        id_counter_start: int,
        exclude_problemspecs: set,
        randomize: bool,
        pbar,
    ):
        problems = []
        problems_generated = 0
        id_counter = id_counter_start
        while problems_generated < n_problems:
            # put in function
            if randomize:
                steps = rng.integers(1, max_steps + 1)
            else:
                steps = max_steps
            state = cube3.reset()
            assert isinstance(state, Cube3State)
            for _ in range(steps):
                actions, _ = cube3.actions_unpruned(state)
                random_action = rng.choice(actions)
                state = cube3.result(state, random_action)

            if state in exclude_problemspecs or cube3.is_goal(state):
                continue
            else:
                exclude_problemspecs.add(state)
                problem = {
                    "front": state.front.tolist(),
                    "up": state.up.tolist(),
                    "down": state.down.tolist(),
                    "left": state.left.tolist(),
                    "right": state.right.tolist(),
                    "back": state.back.tolist(),
                    "id": f"{id_prefix}_{id_counter}",
                }
                problems.append(problem)
                problems_generated += 1
                id_counter += 1
                pbar.update(1)
        return problems

    curriculum_problems = []
    num_curriculum_problems = args.n_problems_per_difficulty * len(curriculum)
    with tqdm.tqdm(total=num_curriculum_problems) as pbar:
        pbar.set_description("Curriculum problems")
        for diff, ms in enumerate(curriculum):
            ms = int(ms)
            this_curr_problems = generate_step_problems(
                args.n_problems_per_difficulty,
                ms,
                f"c{diff}",
                0,
                exclude_problemspecs,
                args.randomize_curriculum_steps,
                pbar,
            )
            curriculum_problems.extend(this_curr_problems)

    trainset_dict = copy(problemset_dict)
    trainset_dict["is_curriculum"] = True
    trainset_dict["randomize_steps"] = args.randomize_curriculum_steps
    trainset_dict["curriculum"] = curriculum
    trainset_dict["problems_per_difficulty"] = args.n_problems_per_difficulty
    trainset_dict["bootstrap_problems"] = bootstrap_problems
    trainset_dict["curriculum_problems"] = curriculum_problems
    trainset_dict["permutation_problems"] = []
    save_problemset(trainset_dict, "train")

    if args.n_valid > 0:
        with tqdm.tqdm(total=args.n_valid) as pbar:
            pbar.set_description("Valid problems")
            valid_problems = generate_step_problems(
                args.n_valid,
                args.test_steps,
                "t",
                0,
                exclude_problemspecs,
                args.randomize_test_steps,
                pbar,
            )
            problemset_dict["problems"] = valid_problems
            problemset_dict["randomize_steps"] = args.randomize_test_steps
        save_problemset(
            problemset_dict,
            "valid",
        )

    if args.n_test > 0:
        with tqdm.tqdm(total=args.n_test) as pbar:
            pbar.set_description("Test problems")
            test_problems = generate_step_problems(
                args.n_test,
                args.test_steps,
                "t",
                0,
                exclude_problemspecs,
                args.randomize_test_steps,
                pbar,
            )
            problemset_dict["problems"] = test_problems
            problemset_dict["randomize_steps"] = args.randomize_test_steps
        save_problemset(
            problemset_dict,
            "test",
        )


if __name__ == "__main__":
    main()
