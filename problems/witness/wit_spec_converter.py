import argparse
from pathlib import Path
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

    problem_specs_old = [
        line_list
        for lines in args.input_path.read_text().split("\n\n")
        if len(line_list := lines.splitlines()) == 4
    ]

    problemset = {
        "domain_name": "Witness",
        "domain_module": "witness",
        "max_num_colors": 4,
        "width": 4,
        "problems": [],
    }

    for old_spec in problem_specs_old:
        new_spec = {}

        new_spec = {
            "init": old_spec[1].replace("Init: ", "").split(" "),
            "goal": old_spec[2].replace("Goal: ", "").split(" "),
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
