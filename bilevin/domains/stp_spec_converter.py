import argparse
from math import sqrt
from pathlib import Path
import pickle as pkl

import numpy as np
import tqdm

from domains.stp import SlidingTilePuzzle, SlidingTilePuzzleState
from search.problem import Problem


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-path",
        type=lambda p: Path(p).absolute(),
        help="path of file with problem instances to read (old spec)",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        type=lambda p: Path(p).absolute(),
        help="path of file to write problem instances (new spec)",
    )

    parser.add_argument(
        "--input-format",
        "-f",
        type=str,
        choices=["lelis", "felner", "korf"],
        help="path of file to write problem instances (new spec)",
    )

    args = parser.parse_args()

    problem_specs_old = args.input_path.read_text().splitlines()
    if args.input_format in ("felner", "korf"):
        problem_specs_old = [
            [int(x) for x in l.split()[1:]] for l in problem_specs_old if l
        ]
        if args.input_format == "felner":
            problem_specs_old.pop()
    else:
        problem_specs_old = [
            [int(x) for x in l.split()] for l in problem_specs_old if l
        ]

    width = int(sqrt(len(problem_specs_old[0])))

    problemset = {
        "domain_name": "SlidingTilePuzzle",
        "problems": [],
    }

    for i, old_spec in tqdm.tqdm(enumerate(problem_specs_old)):
        tiles = np.array(old_spec).reshape(width, width)
        blank_row, blank_col = np.where(tiles == 0)
        blank_row = blank_row.item()
        blank_col = blank_col.item()
        stp = SlidingTilePuzzle(SlidingTilePuzzleState(tiles, blank_row, blank_col))
        problem = Problem(id=i, domain=stp)
        problemset["problems"].append(problem)

    problemset["problems"] = [problemset["problems"]]
    with args.output_path.open("wb") as f:
        pkl.dump(problemset, f)


if __name__ == "__main__":
    main()
