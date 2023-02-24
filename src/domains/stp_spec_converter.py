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
