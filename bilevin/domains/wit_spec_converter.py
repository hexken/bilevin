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
import json
import tqdm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-path",
        type=lambda p: Path(p).absolute(),
        help="path of file with all problem instances to read (old spec), or directory containing a single problem per file",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=lambda p: Path(p).absolute(),
        help="path of file to write problem instances (new spec)",
    )
    parser.add_argument(
        "--puzzle",
        type=str,
        choices=["colors", "triangles"],
        help="witness puzzle type",
    )

    args = parser.parse_args()

    if args.input_path.is_file():
        problem_specs_old = [
            (args.input_path.name, line_list)
            for lines in args.input_path.read_text().split("\n\n")
            if len(line_list := lines.splitlines()) == 4
        ]
    elif args.input_path.is_dir():
        problem_specs_old = []
        for problem_file in args.input_path.iterdir():
            if problem_file.is_file():
                problem_specs_old.extend(
                    [
                        (problem_file.name, line_list)
                        for lines in problem_file.read_text().split("\n\n")
                        if len(line_list := lines.splitlines()) == 4
                    ]
                )
    else:
        raise ValueError(f"Input path {args.input_path} is not a file or directory")

    if args.puzzle == "colors":
        colors = 4
    elif args.puzzle == "triangles":
        colors = 3
    else:
        raise ValueError(f"Invalid puzzle type {args.puzzle}")

    problemset_dict_template = {
        "domain_module": "witness",
        "domain_name": "Witness",
        "width": width,
        "num_actions": 4,
        "max_num_colors": 4 if puzzle == "colors" else 0,
        "in_channels": in_channels,
        "state_t_width": state_t_width,
        "seed": args.seed,
        "puzzle": "colors",
    }

    for i, (prefix, old_spec) in tqdm.tqdm(enumerate(problem_specs_old)):
        new_spec = {}

        init = old_spec[1].replace("Init: ", "").split(" ")
        goal = old_spec[2].replace("Goal: ", "").split(" ")
        new_spec = {
            "init": (int(init[0]), int(init[1])),
            "goal": (int(goal[0]), int(goal[1])),
            "id": f"{prefix}_{i}",
        }
        colored_cells = []
        values = old_spec[3].replace("Colors: |", "").split("|")
        for t in values:
            if not t:
                break
            numbers = t.split(" ")
            colored_cells.append(
                f"{int(numbers[0])} {int(numbers[1])} {int(numbers[2])}"
            )
        new_spec["colored_cells"] = colored_cells

        problemset["problems"].append(new_spec)

    with args.output_path.open("w") as f:
        json.dump(problemset, f, indent=2)


if __name__ == "__main__":
    main()
