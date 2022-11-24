import argparse
import numpy as np
import copy
from pathlib import Path
import random
import tqdm
from domains.sliding_tile_puzzle import SlidingTilePuzzle


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
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

    args = parser.parse_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    problems = []

    tiles = [i for i in range(0, args.width**2)]
    goal = SlidingTilePuzzle(tiles)

    # generating training instances
    for _ in tqdm.tqdm(range(args.num_problems)):
        state = copy.deepcopy(goal)

        steps = random.randint(args.min_steps, args.max_steps)
        for _ in range(steps):
            actions = state.successors()
            random_index = random.randint(0, len(actions) - 1)
            random_action = actions[random_index]
            state.apply_action(random_action)

        problems.append(copy.deepcopy(state))

    with args.output_path.open("w") as f:
        for problem in problems:
            f.write(f"{problem.one_line()}\n")

    # generating test instances
    # j = 0
    # while len(test_instances) < ntest:
    #     tiles = [i for i in range(0, width * width)]
    #     np.random.shuffle(tiles)

    #     state = SlidingTilePuzzle(tiles)
    #     if (
    #         state.is_valid()
    #         and state not in test_instances
    #         and state not in train_instances
    #     ):
    #         state.save_state(join(parameters.file_test, "puzzle_" + str(j + 1)))
    #         test_instances.add(copy.deepcopy(state))
    #         j += 1


if __name__ == "__main__":
    main()
