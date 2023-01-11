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
        "-n",
        "--num-problems",
        type=int,
        default=1000,
        help="number of training puzzles to be generated",
    )

    parser.add_argument(
        "-m",
        "--min-steps",
        type=int,
        default=10,
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

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    problemset = {
        "domain_module": "stp",
        "domain_name": "SlidingTilePuzzle",
        "width": args.width,
        "problems": [],
    }
    problems = set()

    random.seed(args.seed)
    np.random.seed(args.seed)

    tiles = np.arange(args.width**2).reshape(args.width, args.width)
    stp = SlidingTilePuzzle(tiles, tiles)

    # generating training instances
    with tqdm.tqdm(total=args.num_problems) as pbar:
        problem_id = 0
        while len(problems) < args.num_problems:
            state = stp.reset()

            steps = random.randint(args.min_steps, args.max_steps)
            for _ in range(steps):
                actions = stp.actions_unpruned(state)
                random_index = random.randint(0, len(actions) - 1)
                random_action = actions[random_index]
                state = stp.result(state, random_action)

            if state in problems or stp.is_goal(state):
                continue
            else:
                problems.add(state)
                problem = {"tiles": state.tiles.tolist(), "id": problem_id}
                problemset["problems"].append(problem)
                problem_id += 1
                pbar.update(1)

    with args.output_path.open("w") as f:
        json.dump(problemset, f, indent=2)


if __name__ == "__main__":
    main()
