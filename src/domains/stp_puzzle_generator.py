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


def is_valid(tiles):
    """
    Check if a sliding tile puzzle is solvable.
    For nxn grids.
    if n is odd, then the number of inversions must be even.
    if n is even, then the number of inversions + blank_row (0-indexed) must be even.
    """
    blank_row = np.where(tiles == 0)[0].item()
    num_inversions = 0
    width = tiles.shape[0]
    n = width**2
    tiles = tiles.reshape(-1)
    for i in range(0, n):
        for j in range(i + 1, n):
            if tiles[i] and tiles[j] and tiles[i] > tiles[j]:
                num_inversions += 1

    if width % 2 == 1:
        return num_inversions % 2 == 0
    else:
        return (blank_row + num_inversions) % 2 == 0


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
        "--curriculum",
        nargs="+",
        default=[50, 100, 250, 500, 1000],
        help="list of steps away from goal for the curriculum",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )

    args = parser.parse_args()

    if args.n_train <= len(args.curriculum) + 2:
        raise ValueError("n_train must be grater than curriculum_steps + 2")

    args.output_path.mkdir(parents=True, exist_ok=True)

    exclude_problemspecs = set()
    for pset_path in args.exclude_problemset:
        problemset_dict = json.load(pset_path.open("r"))
        domain_module = getattr(domains, problemset_dict["domain_module"])
        problemset = getattr(domain_module, "parse_problemset")(problemset_dict)

        for problem in problemset["problems"]:
            exclude_problemspecs.add(problem.domain.initial_state)

    problem_specs = set()

    rng = np.random.default_rng(args.seed)
    goal_tiles = np.arange(args.width**2).reshape(args.width, args.width)
    stp = SlidingTilePuzzle(goal_tiles, goal_tiles)

    problemset_template = {
        "domain_module": "stp",
        "domain_name": "SlidingTilePuzzle",
        "width": args.width,
    }

    problems = []

    def generate_permutation_problems(num_problems, desc):
        with tqdm.tqdm(total=num_problems) as pbar:
            pbar.set_description(desc)
            generated = 0
            problem_id = len(problem_specs)
            while generated < num_problems:
                tiles = rng.permutation(args.width**2).reshape(
                    (args.width, args.width)
                )
                stp = SlidingTilePuzzle(tiles, goal_tiles)
                state = stp.initial_state
                if (
                    not is_valid(state.tiles)
                    or state in problem_specs
                    or state in exclude_problemspecs
                    or stp.is_goal(state)
                ):
                    continue
                else:
                    problem_specs.add(state)
                    problem = {"tiles": state.tiles.tolist(), "id": problem_id}
                    problems.append(problem)
                    problem_id += 1
                    generated += 1
                    pbar.update(1)

    def save_problemset(problems, suffix):
        problemset_template["problems"] = problems
        path = args.output_path / f"{len(problems)}-{suffix}.json"
        with path.open("w") as f:
            json.dump(problemset_template, f)

    def get_all_reachable_states(pid, state, step, max_step):
        def helper(state, step, max_step):
            nonlocal pid
            if step >= max_step:
                return
            stp = SlidingTilePuzzle(state.tiles, goal_tiles)
            actions = stp.actions_unpruned(state)
            for action in actions:
                new_state = stp.result(state, action)
                if (
                    new_state in problem_specs
                    or new_state in exclude_problemspecs
                    or stp.is_goal(new_state)
                ):
                    continue
                else:
                    problem_specs.add(new_state)
                    problem = {"tiles": new_state.tiles.tolist(), "id": pid}
                    pid += 1
                    problems.append(problem)
                    helper(new_state, step + 1, max_step)

        helper(state, step, max_step)

    if args.n_train > 0:
        get_all_reachable_states(0, stp.initial_state, 0, 10)
        problem_id = len(problems)

        curriculum = args.curriculum

        # split remaining train between specified curriculum and permutations
        num_remaining_problems = args.n_train - len(problems)
        problems_per_difficulty = num_remaining_problems // (len(curriculum) + 1)
        num_curriculum_problems = num_remaining_problems - (
            problems_per_difficulty * len(curriculum)
        )

        if len(problems) > num_curriculum_problems:
            raise ValueError(
                "Num curriculum problems must be greater than all reachable states"
            )

        with tqdm.tqdm(total=num_curriculum_problems) as pbar:
            pbar.set_description("Curriculum problems")
            difficulty = 0
            steps = int(curriculum[difficulty])
            while len(problem_specs) < num_curriculum_problems:
                if problems and len(problems) % problems_per_difficulty == 0:
                    difficulty += 1
                    steps = int(curriculum[difficulty])

                state = stp.reset()
                for _ in range(steps):
                    actions = stp.actions_unpruned(state)
                    random_action = rng.choice(actions)
                    state = stp.result(state, random_action)

                if (
                    state in problem_specs
                    or state in exclude_problemspecs
                    or stp.is_goal(state)
                ):
                    continue
                else:
                    problem_specs.add(state)
                    problem = {"tiles": state.tiles.tolist(), "id": problem_id}
                    problems.append(problem)
                    problem_id += 1
                    pbar.update(1)

            num_remaining_problems -= 
        generate_permutation_problems(
            , "Hard curriculum problems"
        )

        problemset_template["problems_per_difficulty"] = problems_per_difficulty
        save_problemset(problems, "train")
        del problemset_template["problems_per_difficulty"]

    if args.n_valid > 0:
        generate_permutation_problems(args.n_valid, "Validation problems")
        save_problemset(
            problems[args.n_train :],
            "valid",
        )

    if args.n_train > 0:
        generate_permutation_problems(args.n_test, "Test problems")
        save_problemset(
            problems[args.n_train + args.n_valid :],
            "test",
        )


if __name__ == "__main__":
    main()
