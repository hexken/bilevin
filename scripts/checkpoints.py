from pathlib import Path
import re
import sys

from natsort import natsorted


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python checkpoints.py <runs_dir> <dom|all> <min_epoch> <outfile>")
        sys.exit(1)

    if sys.argv[2] == "all":
        dom_dirs = list(Path(sys.argv[1]).glob("*/"))
    else:
        dom_dirs = [Path(sys.argv[1]) / sys.argv[2]]

    min_epoch = sys.argv[3]
    if not min_epoch.isdigit():
        raise ValueError("min_epoch must be an integer")

    s = re.compile(f".*_e{min_epoch}\.pkl$")
    s2 = re.compile(".*train_(.*)_(\d+)_\d+-\d+$")
    outfile = Path(sys.argv[-1]).open("w")

    for domdir in dom_dirs:
        for run in natsorted(domdir.glob("*/*/")):
            good_run = False
            for search_data in run.glob("search*.pkl"):
                if s.match(search_data.name) is not None:
                    good_run = True
                    break

            if not good_run:
                print(run.name)
                m = s2.match(run.name)
                agent = m.group(1)
                seed = m.group(2)
                chkpt = natsorted(run.glob("checkpoint*.pkl"))[-1]
                outfile.write(f"{seed} {agent} {chkpt}\n")
