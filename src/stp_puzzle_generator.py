import argparse
import numpy as np
import copy
from pathlib import Path
import random
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

    problems = set()

    random.seed(args.seed)
    np.random.seed(args.seed)

    tiles = np.arange(args.width**2).reshape(args.width, args.width)
    stp_goal = SlidingTilePuzzle(tiles)

    # generating training instances
    with tqdm.tqdm(total=args.num_problems) as pbar:
        while len(problems) < args.num_problems:
            state = stp_goal.reset()

            steps = random.randint(args.min_steps, args.max_steps)
            for _ in range(steps):
                actions = stp_goal.actions_unpruned(state)
                random_index = random.randint(0, len(actions) - 1)
                random_action = actions[random_index]
                state = stp_goal.result(state, random_action)

            state_t = stp_goal.state_tensor(state)
            if state_t in problems:
                continue
            else:
                problems.add(state)
                pbar.update(1)

    with args.output_path.open("w") as f:
        for problem in problems:
            f.write(f"{problem.oneline()}\n")


if __name__ == "__main__":
    main()
