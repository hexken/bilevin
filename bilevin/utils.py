import linecache
from pathlib import Path
import pickle as pkl
import random
import socket
import tracemalloc

from filelock import FileLock
import numpy as np
import numpy as np
import torch as to
import torch.multiprocessing as mp

from search.loaders import AsyncProblemLoader


def get_async_loader(args, problems_path: Path, batch_size: int | None = None):
    with problems_path.open("rb") as f:
        pset_dict = pkl.load(f)
    problems = pset_dict["problems"][0]
    indexer = mp.Value("I", 0)
    indices = mp.Array("I", len(problems))
    loader = AsyncProblemLoader(
        problems,
        indices,
        indexer,
        batch_size=batch_size,
        seed=args.seed,
    )


def get_problems(args, problems_path: Path, batch_size: int | None = None):
    with problems_path.open("rb") as f:
        pset_dict = pkl.load(f)
    problems = pset_dict["problems"][0]
    return problems, pset_dict


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


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    to.manual_seed(seed)
