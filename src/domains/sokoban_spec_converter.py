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
import pathlib
import pickle

import numpy as np
import tqdm

from domains.sokoban import Sokoban, SokobanState, from_string


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-path",
        type=lambda p: Path(p).absolute(),
        help="path of directory contianing problem files, each with instances, to read (old spec)",
    )
    parser.add_argument(
        "--id-prefix",
        type=str,
        default="",
        help="prefix to add to problem id ([prefix]_id)",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        type=lambda p: Path(p).absolute(),
        help="path of file to write problem instances (new spec)",
    )

    args = parser.parse_args()
    if args.id_prefix:
        id_prefix = f"{args.id_prefix}_"
    else:
        id_prefix = ""

    problem_files = sorted(pathlib.Path(args.input_path).glob("*.txt"))
    problemset = []
    for f in tqdm.tqdm(problem_files):
        all_txt = f.read_text()
        all_problem_strings = all_txt.split("\n\n")
        for problem_string in all_problem_strings:
            if not problem_string:
                continue
            num, new_line, problem_string = problem_string.partition("\n")
            problem_id = f"{id_prefix}{int(num[1:])}"
            problem = from_string(problem_string)
            new_spec = {
                "map": problem.map.astype(np.int8).tolist(),
                "man_row": problem.original_man_row,
                "man_col": problem.original_man_col,
                "boxes": problem.original_boxes.astype(np.int8).tolist(),
                "id": problem_id,
            }
            problemset.append(new_spec)

    width = problem.cols
    in_channels = problem.in_channels

    problemset = {
        "domain_name": "Sokoban",
        "domain_module": "sokoban",
        "width": width,
        "problems": problemset,
        "num_actions": 4,
        "in_channels": in_channels,
        "state_t_width": width,
    }

    with args.output_path.open("w") as f:
        json.dump(problemset, f, indent=0)


if __name__ == "__main__":
    main()
