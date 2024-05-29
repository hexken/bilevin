from pathlib import Path
import pickle as pkl

from utils import get_train_data, get_test_data
import sys


def prepare_domain(domain_pth, keys, outfile, test=False, model_suffix=""):
    pths = list(Path(domain_pth).glob("*/"))
    print(pths)
    if test:
        all_runs = get_test_data(pths, keys, model_suffix=model_suffix)
    else:
        all_runs = get_train_data(pths, keys, min_valids=1)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("wb") as f:
        pkl.dump(all_runs, f)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python get_runs.py <indir> [all] [<keys ...>|<best|latest>] [train|test] <outfile|outdir>"
        )
        sys.exit(1)
    indir = Path(sys.argv[1])

    if sys.argv[-2].lower() == "train":
        test = False
    elif sys.argv[-2].lower() == "test":
        test = True
        if sys.argv[-3] == "best":
            model_suffix = "best"
        elif sys.argv[-3] == "latest":
            model_suffix = "latest"
        else:
            raise ValueError("Third last argument must be either 'best' or 'last'")
    else:
        raise ValueError("Second last argument must be either 'train' or 'test'")

    if sys.argv[2].lower() == "all":
        keys = sys.argv[3:-2] if len(sys.argv) > 5 else ["agent"]
        domains = list(Path(indir).glob("*/"))
        outdir = Path(sys.argv[-1])
        if test:
            outfiles = [outdir / f"{d.name}_{model_suffix}.pkl" for d in domains]
        else:
            outfiles = [outdir / f"{d.name}.pkl" for d in domains]
        keys = [keys for _ in domains]
        for indir, k, o in zip(domains, keys, outfiles):
            prepare_domain(indir, k, o, test)
    else:
        keys = sys.argv[2:-1] if len(sys.argv) > 4 else ["agent"]
        prepare_domain(indir, keys, Path(sys.argv[-1]), test)
