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
import json
from pathlib import Path

import numpy as np
import tqdm

import domains
from domains import SlidingTilePuzzle


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--output-path",
        type=lambda p: Path(p).absolute(),
        help="path to save problem instances",
    )
    parser.add_argument(
        "-x",
        "--exclude-problemset",
        action="extend",
        nargs="+",
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
    # parser.add_argument(
    #     "--curriculum",
    #     action="store_true",
    #     help="generate training problems of increasing difficulty",
    # )
    # parser.add_argument(
    #     "--hard-test",
    #     action="store_true",
    #     help="generate test/val problems using permutations of the goal state"
    # )
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
        "-m",
        "--min-steps",
        type=int,
        default=50,
        help="min number of steps performed backwards from the goal",
    )

    parser.add_argument(
        "-M",
        "--max-steps",
        type=int,
        default=50000,
        help="max number of steps performed backwards from the goal",
    )
    parser.add_argument(
        "-n",
        "--n-batches",
        type=int,
        default=25,
        help="max number of steps performed backwards from the goal",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )

    args = parser.parse_args()
    if args.n_train % args.n_batches != 0:
        raise ValueError("n_train must be divisible by n_batches.")
    elif args.n_train <= args.n_batches < 1:
        raise ValueError("n_train must be grater than n_batches.")

    args.output_path.mkdir(parents=True, exist_ok=True)

    exclude_problemspecs = set()
    for pset_path in args.exclude_problemset:
        problemset_dict = json.load(pset_path.open("r"))
        domain_module = getattr(domains, problemset_dict["domain_module"])
        (
            parsed_problems,
            num_actions,
            in_channels,
            state_t_width,
            double_backward,
        ) = getattr(domain_module, "load_problemset")(problemset_dict)

        for problem in parsed_problems:
            exclude_problemspecs.add(problem.domain.initial_state)

    problem_specs = set()

    rng = np.random.default_rng(args.seed)
    goal_tiles = np.arange(args.width**2).reshape(args.width, args.width)
    stp = SlidingTilePuzzle(goal_tiles, goal_tiles)

    problems = []

    if args.n_train > 0:
        print("Generating training problems...")
        dups = 0
        max_steps = np.linspace(args.min_steps, args.max_steps, args.n_batches)
        problems_per_difficulty = args.n_train // args.n_batches
        with tqdm.tqdm(total=args.n_train) as pbar:
            problem_id = 0
            difficulty = 0
            steps = int(max_steps[difficulty])
            while len(problem_specs) < args.n_train:
                state = stp.reset()
                for _ in range(steps):
                    actions = stp.actions_unpruned(state)
                    random_action = rng.choice(actions)
                    # random_action = actions[random_index]
                    state = stp.result(state, random_action)

                if (
                    state in problem_specs
                    or state in exclude_problemspecs
                    or stp.is_goal(state)
                ):
                    dups += 1
                    continue
                else:
                    problem_specs.add(state)
                    problem = {"tiles": state.tiles.tolist(), "id": problem_id}
                    problems.append(problem)
                    problem_id += 1
                    pbar.update(1)

                    if len(problems) % problems_per_difficulty == 0:
                        difficulty += 1
                        steps = int(max_steps[difficulty])

        print(f"Duplicate problems encountered: {dups}")

    def is_valid(tiles):
        blank_row = np.where(tiles == 0)[0].item()
        num_inversions = 0
        tiles = state.tiles.reshape(-1)
        n = len(tiles)
        for i in range(0, n):
            for j in range(i + 1, n):
                if tiles[i] > tiles[j]:
                    num_inversions += 1

        if (tiles.shape[0] - blank_row) % 2 == 0:
            return num_inversions % 2 == 1
        else:
            return num_inversions % 2 == 0

    def generate_permutation_problems(num_problems):
        with tqdm.tqdm(total=num_problems) as pbar:
            problem_id = 0
            dups = 0
            while len(problem_specs) < args.n_train:
                tiles = rng.permutation(args.width**2)
                stp = SlidingTilePuzzle(tiles, goal_tiles)
                state = stp.initial_state
                if (
                    state in problem_specs
                    or state in exclude_problemspecs
                    or stp.is_goal(state)
                    or not is_valid(state.tiles)
                ):
                    dups += 1
                    continue
                else:
                    problem_specs.add(state)
                    problem = {"tiles": state.tiles.tolist(), "id": problem_id}
                    problems.append(problem)
                    problem_id += 1
                    pbar.update(1)

            print(f"Duplicate problems encountered: {dups}")

    if args.n_valid > 0:
        print("Generating validation problems...")
        generate_permutation_problems(args.n_valid)

    if args.n_train > 0:
        print("Generating testing problems...")
        generate_permutation_problems(args.n_test)

    problemset_template = {
        "domain_module": "stp",
        "domain_name": "SlidingTilePuzzle",
        "width": args.width,
    }

    for n, suffix in [
        (args.n_train, "train"),
        (args.n_valid, "valid"),
        (args.n_test, "test"),
    ]:
        if n > 0:
            problemset_template["problems"] = problems[:n]
            path = args.output_path / f"{n}-{suffix}.json"
            with path.open("w") as f:
                json.dump(problemset_template, f)
            problems = problems[n:]


if __name__ == "__main__":
    main()
