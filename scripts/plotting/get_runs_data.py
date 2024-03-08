from pathlib import Path
import pickle as pkl

from utils import get_runs_data
import sys


def prepare_domain(domain_pth, keys, outfile):
    pths = list(Path(domain_pth).glob("*/"))
    print(pths)
    all_runs = get_runs_data(pths, keys, min_valids=1)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("wb") as f:
        pkl.dump(all_runs, f)


if __name__ == "__main__":
    indir = Path(sys.argv[1])
    if len(sys.argv) < 3:
        print("Usage: python get_runs.py <indir> [all] [<keys>] <outfile|outdir>")
        sys.exit(1)

    if sys.argv[2].lower() == "all":
        keys = sys.argv[3:-1] if len(sys.argv) > 4 else ["agent"]
        domains = list(Path(indir).glob("*/"))
        outdir = Path(sys.argv[-1])
        outfiles = [outdir / f"{d.name}.pkl" for d in domains]
        keys = [keys for _ in domains]
        for indir, k, o in zip(domains, keys, outfiles):
            prepare_domain(indir, k, o)
    else:
        keys = sys.argv[2:-1] if len(sys.argv) > 3 else ["agent"]
        prepare_domain(indir, keys, Path(sys.argv[-1]))
