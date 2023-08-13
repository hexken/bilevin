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
from collections import deque
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import random

import numpy as np
from tqdm import tqdm

from domains import Witness
from domains.witness import WitnessState


def main():
    """
    Generate a dataset of Witness triangle problems. A generated problem instance is only kept if
    there least args.width // 2 triangles of size at least 2
    """
    parser = argparse.ArgumentParser()

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
        "-p",
        "--triangle-prob",
        type=float,
        default=0.1,
        help="probability of placing a triangle in each snake adjacent cell",
    )
    parser.add_argument(
        "--min-path-ratio",
        type=float,
        default=0.35,
        help="path that generated problem must be >= this ratio of squared width",
    )

    args = parser.parse_args()

    args.output_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    goals = (
        [(0, i) for i in range(args.width + 1)]
        + [(i, 0) for i in range(args.width + 1)]
        + [(args.width, i) for i in range(args.width + 1)]
        + [(i, args.width) for i in range(args.width + 1)]
    )
    goals.remove((0, 0))

    problem_specs = set()

    problems = []

    problem_id = 0
    total_num_problems = args.n_train + args.n_valid + args.n_test

    with tqdm(total=total_num_problems) as pbar:
        prefix = "tr_"
        while len(problem_specs) < total_num_problems:
            if len(problem_specs) == args.n_train:
                prefix = "te_"
                problem_id = 0
            head_at_goal = False
            goal = rng.choice(goals)
            problem = {
                "init": (0, 0),  # todo allow other initial pos?
                "goal": goal.tolist(),
            }
            wit = Witness(
                puzzle="triangles",
                width=args.width,
                max_num_colors=3,
                **problem,
            )
            # generate a path from start to goal
            state = wit.reset()
            assert isinstance(state, WitnessState)
            while True:
                actions, _ = wit.actions_unpruned(state)
                if not actions:
                    break
                action = rng.choice(actions)
                state = wit.result(state, action)
                if wit.is_head_at_goal(state):
                    head_at_goal = True
                    break

            if not head_at_goal:
                continue

            # heuristic to make sure the path is not too short
            if (state.v_segs.sum() + state.h_segs.sum()) / state.grid.shape[
                0
            ] ** 2 < args.min_path_ratio:
                continue

            colored_cells = []
            for row, col in product(range(args.width), range(args.width)):
                n_adj = int(
                    state.v_segs[row, col]
                    + state.v_segs[row, col + 1]
                    + state.h_segs[row, col]
                    + state.h_segs[row + 1, col]
                )

                if n_adj == 0:
                    continue
                elif n_adj == 4:
                    wit.plot(state)
                    raise ValueError("cell has 4 adjacent segments")

                if rng.random() <= args.triangle_prob and n_adj >= 1:
                    colored_cells.append(f"{row} {col} {n_adj}")

            if len(colored_cells) < args.width // 2:
                continue

            problem["colored_cells"] = colored_cells
            problem_str = str(problem)
            if problem_str in problem_specs:
                continue
            else:
                problem_specs.add(problem_str)
                problem["id"] = f"{prefix}{problem_id}"
                problem_id += 1
                problems.append(problem)
                pbar.update()

    assert isinstance(wit, Witness)
    num_actions = wit.num_actions
    in_channels = wit.in_channels
    width = wit.width
    state_t_width = wit.state_width

    problemset_dict_template = {
        "domain_module": "witness",
        "domain_name": "Witness",
        "width": width,
        "num_actions": num_actions,
        "max_num_colors": 3,
        "in_channels": in_channels,
        "state_t_width": state_t_width,
        "seed": args.seed,
        "puzzle": "triangles",
    }

    for n, suffix in [
        (args.n_train, "train"),
        (args.n_valid, "valid"),
        (args.n_test, "test"),
    ]:
        problemset_dict = deepcopy(problemset_dict_template)
        if n == 0:
            continue
        elif suffix == "train":
            problemset_dict["is_curriculum"] = True
            problemset_dict["bootstrap_problems"] = []
            problemset_dict["permutation_problems"] = []

            train_problems = problems[:n]
            problemset_dict["curriculum_problems"] = train_problems
            problemset_dict["curriculum"] = ["unfiltered"]
            problemset_dict["problems_per_difficulty"] = len(train_problems)
        else:
            problemset_dict["problems"] = problems[:n]
        path = args.output_path / f"{n}-{suffix}.json"
        with path.open("w") as f:
            json.dump(problemset_dict, f)
        problems = problems[n:]


if __name__ == "__main__":
    main()
