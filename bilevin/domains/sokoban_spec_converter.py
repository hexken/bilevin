import argparse
from collections import deque
from heapq import heappop, heappush
import os
from pathlib import Path
import pathlib
import pickle as pkl
import sys

import numpy as np
import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from domains.sokoban import Sokoban, SokobanState
from search.loaders import Problem


def empty_neighbours(r, c, map):
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = r + dr, c + dc
        if (
            map[Sokoban.wall_channel, rr, cc] == 0
            and map[Sokoban.box_goal_channel, rr, cc] == 0
        ):
            yield (rr, cc)


def count_reachable_positions(map):
    candidates = set()
    for r, c in zip(*np.where(map[Sokoban.box_goal_channel])):
        for pos in empty_neighbours(r, c, map):
            candidates.add(pos)

    reachables = []
    for cand in candidates:
        count = 0
        frontier = deque([cand])
        visited = {cand}
        while len(frontier) > 0:
            pos = frontier.popleft()
            count += 1
            r, c = pos
            for new_pos in empty_neighbours(r, c, map):
                if new_pos not in visited:
                    frontier.append(new_pos)
                    visited.add(new_pos)
        heappush(reachables, (-count, cand))
    return reachables


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-path",
        type=lambda p: Path(p).absolute(),
        help="path of directory contianing problem files, each with instances, to read (old spec)",
    )
    parser.add_argument(
        "--id-start",
        type=int,
        default=0,
        help="starting index for problem ids",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="save as a test problemset (default is train)",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=lambda p: Path(p).absolute(),
        help="path of file to write problem instances (new spec)",
    )

    args = parser.parse_args()

    problem_files = sorted(pathlib.Path(args.input_path).glob("*.txt"))
    problems = []
    id_counter = args.id_start
    for f in tqdm.tqdm(problem_files):
        all_txt = f.read_text()
        all_problem_strings = all_txt.split(";")
        for problem_string in all_problem_strings[:5]:
            if len(problem_string) < 20:
                continue
            lines = problem_string.splitlines()
            lines = lines[1:-1]
            print("orignial string")
            print("\n".join(lines))

            rows = len(lines)
            cols = len(lines[0])
            map = np.zeros((3, rows, cols), dtype=np.float32)
            boxes = np.zeros((rows, cols), dtype=np.float32)
            man_row = -1
            man_col = -1

            for i in range(rows):
                for j in range(cols):
                    if lines[i][j] == Sokoban.goal_str:
                        map[Sokoban.box_goal_channel, i, j] = 1

                    if lines[i][j] == Sokoban.man_str:
                        man_row = i
                        man_col = j

                    if lines[i][j] == Sokoban.wall_str:
                        map[Sokoban.wall_channel, i, j] = 1

                    if lines[i][j] == Sokoban.box_str:
                        boxes[i, j] = 1
            assert 1 <= man_row <= rows - 2
            assert 1 <= man_col <= cols - 2
            assert (boxes == 1).sum() == 4

            state = SokobanState(man_row, man_col, boxes)
            # todo set man_goal_channel
            reachables = count_reachable_positions(map)
            r, c = reachables[0][1]
            print(reachables)
            map[Sokoban.man_goal_channel, r, c] = 1

            domain = Sokoban(state, map, forward=True)
            print("converted string")
            domain.print(state)
            problem = Problem(id=id_counter, domain=domain)
            problems.append(problem)
            id_counter += 1

    problemset = {
        "domain_name": "Sokoban",
        "problems": problems,
    }
    problemset["problems"] = [problemset["problems"]]
    with args.output_path.open("wb") as f:
        pkl.dump(problemset, f)


if __name__ == "__main__":
    main()
