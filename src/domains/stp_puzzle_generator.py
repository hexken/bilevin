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
import random

import numpy as np
import tqdm

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
        default=1000,
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

    args.output_path.mkdir(parents=True, exist_ok=True)

    problem_specs = set()

    random.seed(args.seed)
    np.random.seed(args.seed)

    tiles = np.arange(args.width**2).reshape(args.width, args.width)
    stp = SlidingTilePuzzle(tiles, tiles)

    total_num_problems = args.n_train + args.n_valid + args.n_test
    problems = []

    with tqdm.tqdm(total=total_num_problems) as pbar:
        problem_id = 0
        while len(problem_specs) < total_num_problems:
            state = stp.reset()

            steps = random.randint(args.min_steps, args.max_steps)
            for _ in range(steps):
                actions = stp.actions_unpruned(state)
                random_index = random.randint(0, len(actions) - 1)
                random_action = actions[random_index]
                state = stp.result(state, random_action)

            if state in problem_specs or stp.is_goal(state):
                print("duplicate")
                continue
            else:
                problem_specs.add(state)
                problem = {"tiles": state.tiles.tolist(), "id": problem_id}
                problems.append(problem)
                problem_id += 1
                pbar.update(1)

    problemset = {
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
            problemset["problems"] = problems[:n]
            path = args.output_path / f"{n}-{suffix}.json"
            with path.open("w") as f:
                json.dump(problemset, f)
            problems = problems[n:]


if __name__ == "__main__":
    main()
