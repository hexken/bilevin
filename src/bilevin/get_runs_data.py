from pathlib import Path
import pickle as pkl

from plotting.utils import get_runs_data
import sys

if __name__ == "__main__":
    outfile = Path(sys.argv[-1])
    outfile.parent.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) < 3:
        print("Usage: python get_runs.py <indir> [<keys>] <outfile>")
        sys.exit(1)

    pths = list(Path(sys.argv[1]).glob("*/"))
    print(pths)
    keys = sys.argv[2:-1] if len(sys.argv) > 3 else ["agent"]
    all_runs = get_runs_data(pths, keys, min_valids=1)
    with outfile.open("wb") as f:
        pkl.dump(all_runs, f)
