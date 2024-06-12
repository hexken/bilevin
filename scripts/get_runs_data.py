from pathlib import Path
import pickle as pkl

from plotting.utils import get_train_data, get_test_data
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
            "Usage: python get_runs.py <indir> [all] [train|test] [best|latest] <outfile|outdir>"
        )
        sys.exit(1)
    indir = Path(sys.argv[1])

    if sys.argv[-3].lower() == "train":
        test = False
    elif sys.argv[-3].lower() == "test":
        test = True
        pass
        if sys.argv[-2] == "best":
            model_suffix = "best"
        elif sys.argv[-2] == "latest":
            model_suffix = "latest"
        else:
            raise ValueError("Third last argument must be either 'best' or 'last'")
    else:
        raise ValueError("Second last argument must be either 'train' or 'test'")

    keys = ["agent"]
    if sys.argv[2].lower() == "all":
        domains = list(Path(indir).glob("*/"))
        outdir = Path(sys.argv[-1])
        if test:
            outfiles = [outdir / f"{d.name}.pkl" for d in domains]
        else:
            outfiles = [outdir / f"{d.name}.pkl" for d in domains]
        keys = [keys for _ in domains]
        for indir, k, o in zip(domains, keys, outfiles):
            prepare_domain(indir, k, o, test)
    else:
        prepare_domain(indir, keys, Path(sys.argv[-1]), test)
