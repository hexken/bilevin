import linecache
from pathlib import Path
import pickle as pkl
import socket
import tracemalloc

from filelock import FileLock
import numpy as np


def display_top(snapshot, key_type="lineno", limit=25):
    """
    from https://docs.python.org/3/library/tracemalloc.html
    """
    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        )
    )
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print(
            "#%s: %s:%s: %.1f KiB"
            % (index, frame.filename, frame.lineno, stat.size / 1024)
        )
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print("    %s" % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def find_free_port(lockfile: str, master_addr: str) -> str:

    def _find_free_port():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((master_addr, 0))
        _, port = sock.getsockname()
        sock.close()
        return port

    portfile = Path(f"{lockfile}.pkl")
    lockfile = f"{lockfile}.lock"
    lock = FileLock(lockfile)
    with lock:
        if portfile.is_file():
            f = portfile.open("r+b")
            ports = pkl.load(f)
            while True:
                port = _find_free_port()
                if port in ports:
                    continue
                else:
                    break
            f.seek(0)
            ports.add(port)
            pkl.dump(ports, f)
            f.truncate()
            f.close()
        else:
            port = _find_free_port()
            ports = {port}
            with portfile.open("wb") as f:
                pkl.dump(ports, f)
    return str(port)


def split_by_rank(args, problems):
    "split a list of lists of problems into a list of lists of problems per rank"
    rng = np.random.default_rng(args.seed)

    def split_by_rank(problems):
        ranks_x_problems = [[] for _ in range(args.world_size)]
        rank = 0
        for problem in problems:
            ranks_x_problems[rank].append(problem)
            rank = (rank + 1) % args.world_size

        # ensure all ranks have same number of problems per stage
        n_largest_pset = len(ranks_x_problems[0])
        for pset in ranks_x_problems:
            if len(pset) < n_largest_pset:
                pset.append(rng.choice(problems))
            assert len(pset) == n_largest_pset

        return ranks_x_problems

    stages_x_problems = problems
    num_stages = len(stages_x_problems)

    # turn stages x problems into stages x ranks x problems
    stages_x_ranks_x_problems = []
    for stage_problems in stages_x_problems:
        stages_x_ranks_x_problems.append(split_by_rank(stage_problems))

    world_num_problems = 0
    ranks_x_stages_x_problems = []
    for rank in range(args.world_size):
        curr_stages_x_problems = []
        for stage in range(num_stages):
            probs = stages_x_ranks_x_problems[stage][rank]
            curr_stages_x_problems.append(probs)
            world_num_problems += len(probs)
        ranks_x_stages_x_problems.append(curr_stages_x_problems)

    return ranks_x_stages_x_problems, world_num_problems
