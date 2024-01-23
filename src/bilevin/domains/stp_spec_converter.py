import argparse
from pathlib import Path
import tqdm
import numpy as np
import json
from math import sqrt


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
        "domain_module": "stp",
        "problems": [],
        "num_actions": 4,
        "in_channels": int(width**2),
        "width": width,
        "state_t_width": width,
    }

    for i, old_spec in tqdm.tqdm(enumerate(problem_specs_old)):
        new_spec = {}
        tiles = np.array(old_spec).reshape(width, width).tolist()
        new_spec = {
            "tiles": tiles,
            "id": i,
        }

        problemset["problems"].append(new_spec)

    with args.output_path.open("w") as f:
        json.dump(problemset, f, indent=0)


if __name__ == "__main__":
    main()
