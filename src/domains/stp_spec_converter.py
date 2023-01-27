import argparse
from pathlib import Path
import tqdm
import numpy as np
import json


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

    args = parser.parse_args()

    problem_specs_old = args.input_path.read_text().split(" \n")
    problem_specs_old = [[int(x) for x in l.split(" ")] for l in problem_specs_old if l]

    problemset = {
        "domain_name": "SlidingTilePuzzle",
        "domain_module": "stp",
        "max_num_colors": 4,
        "width": 5,
        "problems": [],
    }
    width = problemset["width"]

    for i, old_spec in tqdm.tqdm(enumerate(problem_specs_old)):
        new_spec = {}
        tiles = np.array(old_spec).reshape(width, width).tolist()
        new_spec = {
            "tiles": tiles,
            "id": i,
        }

        problemset["problems"].append(new_spec)

    with args.output_path.open("w") as f:
        json.dump(problemset, f, indent=2)


if __name__ == "__main__":
    main()
