from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import pickle as pkl
import sys
from typing import Iterable

from natsort import natsorted
import pandas as pd

from plotting.utils import allowable_domains


def process_train_run(agent: str, run_paths: list[Path]):

    i = 0
    runs = []
    for path in run_paths:
        try:
            assert (path / "training_completed.txt").exists()
            run_train_dfs = []
            for p in natsorted(path.glob("train_e*.pkl")):
                df = pkl.load(p.open("rb"))
                df["epoch"] = int(p.stem.split("e")[-1])
                run_train_dfs.append(df)
            run_train_df = pd.concat(run_train_dfs, ignore_index=True)

            run_valid_dfs = []
            for p in natsorted(path.glob("valid_e*.pkl")):
                df = pkl.load(p.open("rb"))
                df["epoch"] = int(p.stem.split("e")[-1])
                run_valid_dfs.append(df)
            run_valid_df = pd.concat(run_valid_dfs, ignore_index=True)

            test_df = pkl.load((path / "test.pkl").open("rb"))

            data = {}
            data["path"] = path.name
            data["args"] = json.load((path / "args.json").open("r"))
            data["train"] = run_train_df
            data["valid"] = run_valid_df
            data["test"] = test_df
            runs.append(data)
            i += 1
        except Exception as e:
            print(f"Error loading {path}: {e}")
    print(f"Loaded {i}/{len(run_paths)} runs of {agent}")

    return runs


def get_train_data(
    pths: Path | Iterable[Path],
) -> dict:
    # get list of paths for each run_name, specified by group_key (should correpsond to seeds)
    if isinstance(pths, Path):
        pths = [pths]

    all_runs = {}
    for agent_pth in pths:
        agent = agent_pth.name
        run_paths = []
        for seed_pth in agent_pth.glob("*/"):
            run_paths.append(seed_pth)
        all_runs[agent] = run_paths

    with ProcessPoolExecutor() as executor:
        future_to_run = {
            executor.submit(process_train_run, agent, run_paths): (
                agent,
                run_paths,
            )
            for agent, run_paths in all_runs.items()
        }
        for f in as_completed(future_to_run):
            try:
                agent, run_paths = future_to_run[f]
                agent_runs = f.result()
                all_runs[agent] = agent_runs
            except Exception as exc:
                print(f"Run {agent} generated an exception: {exc}")

    return all_runs


def prepare_domain(domain_pth, outfile):
    pths = list(Path(domain_pth).glob("*/"))
    print(f"{domain_pth.stem} has {len(pths)} agents")
    all_runs = get_train_data(pths)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("wb") as f:
        pkl.dump(all_runs, f)
    print(f"Saved {domain_pth.stem} to {outfile}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python get_runs.py <indir> <outdir>")
        sys.exit(1)
    indir = Path(sys.argv[1])

    domains = [
        p for p in natsorted(Path(indir).glob("*/")) if p.name in allowable_domains
    ]
    outdir = Path(sys.argv[2])
    outfiles = [outdir / f"{d.name}.pkl" for d in domains]
    for indir, o in zip(domains, outfiles):
        prepare_domain(indir, o)
