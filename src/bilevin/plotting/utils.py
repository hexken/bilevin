from collections import OrderedDict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import pickle as pkl
import re
import subprocess

import matplotlib as mpl
from natsort import natsorted
import pandas as pd

import plotting.keys as pkeys


class RunSingle:
    def __init__(self, path: Path, keys: list[str]):
        self.path = path
        self.allargs = json.load((path / "args.json").open("rb"))

        self.keys = keys
        self.keyargs = {}
        for k in keys:
            self.keyargs[k] = self.allargs[k]

        self.name = " ".join([str(self.keyargs[k]) for k in self.keys])
        self.key = str(self.keyargs)

    def args_key(self, keys):
        return " ".join((str(self.allargs[k]) for k in keys))


class RunSeeds:
    def __init__(self, runs: list[RunSingle], data: dict):
        self.runs = runs
        self.data = data
        self.paths = [r.path for r in runs]
        self.allargs = runs[0].allargs.copy()
        self.keys = runs[0].keys.copy()
        self.keyargs = runs[0].keyargs.copy()
        self.name = runs[0].name
        self.key = str(self.keyargs)

    def args_key(self, keys: list[str]):
        return self.runs[0].args_key(keys)

    def __len__(self):
        return len(self.paths)


def process_run(run_name: str, runs: list[RunSingle], batch_size=4, min_valids=25):
    batch_regex = re.compile(".*_b(\d+).pkl")
    train_dfs = []
    search_dfs = []
    valid_dfs = []
    curr_end_batches = []
    curr_end_times = []
    for r in runs:
        rp = r.path
        run_train_curr_dfs = [
            pkl.load(p.open("rb"))
            for p in natsorted(rp.glob("model_train*.pkl"))
            if "_e" not in p.name
        ]

        run_train_fs_dfs = [
            pkl.load(p.open("rb")) for p in natsorted(rp.glob("model_train*_e*.pkl"))
        ]

        run_search_curr_dfs = [
            pkl.load(p.open("rb"))
            for p in natsorted(rp.glob("search_train*.pkl"))
            if "_e" not in p.name
        ]
        run_search_fs_dfs = [
            pkl.load(p.open("rb")) for p in natsorted(rp.glob("search_train*_e*.pkl"))
        ]

        i = 1
        for tdf, sdf in zip(run_train_curr_dfs, run_search_curr_dfs):
            tdf["stage"] = i
            sdf["stage"] = i
            tdf["epoch"] = 1
            sdf["epoch"] = 1
            i += 1
        for j, (tdf, sdf) in enumerate(
            zip(run_train_fs_dfs, run_search_fs_dfs), start=1
        ):
            tdf["stage"] = i
            sdf["stage"] = i
            tdf["epoch"] = j
            sdf["epoch"] = j

        if len(run_train_curr_dfs) == 0 and len(run_train_fs_dfs) == 0:
            print(f"Warning: no train data found for {rp.name}, skipping")
            continue
        run_train_df = pd.concat(
            run_train_curr_dfs + run_train_fs_dfs, ignore_index=True
        )
        run_train_df["batch"] = run_train_df.index + 1

        run_search_df = pd.concat(
            run_search_curr_dfs + run_search_fs_dfs, ignore_index=True
        )
        run_search_df["batch"] = run_search_df.index.map(
            lambda x: (x // batch_size) + 1
        )
        run_search_df["solved"] = run_search_df["len"].map(pd.notna)
        run_search_df = make_exp_col(run_search_df)
        run_search_df = int_cols_to_float(run_search_df)
        curr_end_times.append(
            run_search_df[run_search_df["epoch"] == 1]
            .groupby("batch")["time"]
            .max()
            .sum()
        )
        curr_end_batches.append(
            run_search_df[run_search_df["epoch"] == 1]["batch"].max()
        )

        run_valid_df_paths = natsorted(rp.glob("search_valid*.pkl"))
        run_valid_dfs = [pkl.load(p.open("rb")) for p in run_valid_df_paths]
        batches = [int(batch_regex.match(p.name).group(1)) for p in run_valid_df_paths]

        if len(batches) < min_valids:
            print(f"Warning: only found {len(batches)} valids for {rp.name}, skipping")
            continue

        for df, batch in zip(run_valid_dfs, batches):
            df["batch"] = batch

        run_valid_df = pd.concat(run_valid_dfs, ignore_index=True)
        run_valid_df["solved"] = run_valid_df["len"].map(pd.notna)
        batch_times = run_search_df.groupby("batch")["time"].max()
        run_valid_df["start_time"] = run_valid_df["batch"].map(
            lambda x: batch_times.loc[:x].sum()
        )
        run_valid_df = make_exp_col(run_valid_df)
        run_valid_df = int_cols_to_float(run_valid_df)

        train_dfs.append(run_train_df)
        search_dfs.append(run_search_df)
        valid_dfs.append(run_valid_df)

    print(f"Loaded {len(search_dfs)}/{len(runs)} runs of {run_name}")
    return {
        "train": train_dfs,
        "search": search_dfs,
        "valid": valid_dfs,
        "curr_end_batch": curr_end_batches,
        "curr_end_time": curr_end_times,
    }


def get_runs_data(
    pths: Path | Iterable[Path], keys: list[str], batch_size=4, min_valids=25
) -> dict:
    # get list of paths for each run_name, specified by group_key (should correpsond to seeds)
    if isinstance(pths, Path):
        pths = [pths]

    all_runs = [RunSingle(rp, keys) for pth in pths for rp in pth.glob("*/")]
    all_runs.sort(key=lambda x: x.name)
    grouped_runs = itertools.groupby(all_runs, lambda x: x.name)
    # Prepare sorting by algorithm
    runs_combined = [(name, list(runs)) for name, runs in grouped_runs]
    runs_data = OrderedDict()

    # print(list(*zip(*runs_combined)))
    with ProcessPoolExecutor() as executor:
        future_to_run = {
            executor.submit(process_run, name, runs, batch_size, min_valids): (
                name,
                runs,
            )
            for name, runs in runs_combined
        }
        for f in as_completed(future_to_run):
            try:
                name, runs = future_to_run[f]
                data = f.result()
                runs_data[name] = RunSeeds(runs, data)
            except Exception as exc:
                print(f"Run {name} generated an exception: {exc}")

    return runs_data


def make_exp_col(df: pd.DataFrame) -> pd.DataFrame:
    if "fexp" in df.columns and "bexp" in df.columns:
        df["exp"] = df["fexp"] + df["bexp"]
    else:
        df["exp"] = df["fexp"]
    return df


def int_cols_to_float(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == pd.UInt32Dtype() or df[col].dtype == pd.UInt64Dtype():
            df[col] = df[col].astype(float)
    return df


class PdfTemplate:
    def __init__(self, figsdir: Path, outfile: str, toc=True):
        self.figs = figsdir
        self.toc = toc
        self.outfile = outfile
        self.latex_cmds: list = []
        self.text: str = ""

    def page_sort_key(self, s: str):
        pass

    def render(self):
        self._pagebreak()
        for fig in sorted(self.figs.glob("*.png"), key=lambda x: x.stem):
            self._h1(fig.stem.replace("_", " "))
            self._img(str(fig))
            self._pagebreak()
        self.text = "\n\n".join(self.latex_cmds)

    def export(self):
        md_file = f"{self.outfile}.md"
        pdf_file = f"{self.outfile}.pdf"
        pandoc = ["pandoc", f"{md_file}", "-V geometry:margin=1cm", f"-o{pdf_file}"]
        with open(md_file, "w") as f:
            f.write(self.text)
        if self.toc:
            pandoc.append("--toc")
        subprocess.run(pandoc)

    def _pagebreak(self):
        self.latex_cmds.append("\pagebreak")

    def _h1(self, text):
        self.latex_cmds.append(f"# {text}")

    def _img(self, img):
        self.latex_cmds.append(f"![]({img}){{width=600px height=700px}}")


class LineStyleMapper:
    def __init__(self):
        self.cmap = mpl.colormaps["tab10"]
        self.astar_c = 0
        self.levin_c = 1
        self.phs_c = 2
        self.uni_ls = "-"
        self.bibfs_ls = "--"
        self.bialt_ls = ":"
        self.used_colors = set()

    def get_ls(self, s: str):
        if "AStar" in s:
            c = self.cmap.colors[self.astar_c]
        elif "Levin" in s:
            c = self.cmap.colors[self.levin_c]
        elif "PHS" in s:
            c = self.cmap.colors[self.phs_c]
        else:
            i = 3
            while i in self.used_colors:
                i += 1
            if i >= len(self.cmap.colors):
                c = 0
            else:
                c = self.cmap.colors[i]

        self.used_colors.add(c)

        if "Alt" in s:
            ls = self.bialt_ls
        elif "BFS" in s:
            ls = self.bibfs_ls
        else:
            ls = self.uni_ls

        return c, ls
