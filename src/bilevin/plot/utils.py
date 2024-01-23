from collections import OrderedDict
from collections.abc import Iterable
import itertools
import json
from pathlib import Path
import pickle as pkl
import re
import subprocess

import matplotlib as mpl
from natsort import natsorted
import pandas as pd


# todo use same color for each major alg, different linestyle for minor
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


def all_group_key(pth):
    r = re.search(
        "^.*_((?:Bi)?AStar.*)?((?:Bi)?Levin.*)?((?:Bi)?PHS.*)?_e\d+_t\d+.\d+_(lr\d+\.\d+)(?:_)(w\d+\.?\d*)?.*",
        str(pth),
    )
    if r:
        return " ".join([g for g in r.groups() if g is not None])


def phs_test_key(pth):
    r = re.search(
        "^.*(PHS).*_opt(.*)_(lr\d+\.\d+)_(n[tf])_(mn[-+]?\d\.\d+)_(m\d\.\d)_loss(.*)_\d_.*",
        str(pth),
    )
    if r:
        return " ".join([g for g in r.groups() if g is not None])


def alg_sort_key(s: str):
    if "PHS" in s:
        key1 = 3
    elif "Levin" in s:
        key1 = 2
    elif "AStar" in s:
        key1 = 0
    else:
        raise ValueError("Unknown alg")

    if "Bi" in s:
        key2 = 1
    else:
        key2 = 0

    if "Alt" in s:
        key3 = 0
    elif "BFS" in s:
        key3 = 1
    else:
        key3 = 3

    return key1, key2, key3


def get_runs_data(pth: Path | Iterable[Path], group_key) -> dict:
    batch_regex = re.compile(".*_b(\d+).pkl")
    # get list of paths for each run_name, specified by group_key (should correpsond to seeds)
    if isinstance(pth, Path):
        pth = [pth]
    all_runs_paths = []
    for p in pth:
        all_runs_paths.extend(list(p.glob("*/")))

    runs_names = []
    runs_paths = []
    all_runs_paths = sorted(all_runs_paths, key=group_key)
    for k, g in itertools.groupby(all_runs_paths, group_key):
        runs_names.append(k)
        runs_paths.append(list(g))

    # sort again by algorithm for plotting
    runs_combined = sorted(
        zip(runs_names, runs_paths), key=lambda x: alg_sort_key(x[0])
    )
    runs_names, runs_paths = zip(*runs_combined)

    runs_data = OrderedDict()
    for run_name, run_paths in zip(runs_names, runs_paths):
        print(f"Found {len(run_paths)} runs of {run_name}")
        train_dfs = []
        search_dfs = []
        valid_dfs = []
        for rp in run_paths:
            print(f"Loading {rp.name}")
            run_train_curr_dfs = [
                pkl.load(p.open("rb"))
                for p in natsorted(rp.glob("model_train*.pkl"))
                if "_e" not in p.name
            ]

            run_train_fs_dfs = [
                pkl.load(p.open("rb"))
                for p in natsorted(rp.glob("model_train*_e*.pkl"))
            ]

            run_search_curr_dfs = [
                pkl.load(p.open("rb"))
                for p in natsorted(rp.glob("search_train*.pkl"))
                if "_e" not in p.name
            ]
            run_search_fs_dfs = [
                pkl.load(p.open("rb"))
                for p in natsorted(rp.glob("search_train*_e*.pkl"))
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
                print(f"Warning: no train data found for {rp.name}")
                continue
            run_train_df = pd.concat(
                run_train_curr_dfs + run_train_fs_dfs, ignore_index=True
            )
            run_train_df.index = run_train_df.index + 1
            run_train_df.index = run_train_df.index.rename("batch")

            run_search_df = pd.concat(
                run_search_curr_dfs + run_search_fs_dfs, ignore_index=True
            )
            run_search_df.index = run_search_df.index + 1
            run_search_df.index = run_search_df.index.rename("batch")
            run_search_df["solved"] = run_search_df["len"].map(pd.notna)
            run_search_df = make_exp_col(run_search_df)
            run_search_df = int_cols_to_float(run_search_df)

            run_valid_df_paths = natsorted(rp.glob("search_valid*.pkl"))
            run_valid_dfs = [pkl.load(p.open("rb")) for p in run_valid_df_paths]
            batches = [
                int(batch_regex.match(p.name).group(1)) for p in run_valid_df_paths
            ]

            if len(batches) == 0:
                print(f"Warning: no valid data found for {rp.name}")
                continue

            for df, batch in zip(run_valid_dfs, batches):
                df["batch"] = batch

            run_valid_df = pd.concat(run_valid_dfs, ignore_index=True)
            run_valid_df["solved"] = run_valid_df["len"].map(pd.notna)
            run_valid_df = make_exp_col(run_valid_df)
            run_valid_df = int_cols_to_float(run_valid_df)

            train_dfs.append(run_train_df)
            search_dfs.append(run_search_df)
            valid_dfs.append(run_valid_df)

        runs_data[run_name] = {
            "train": train_dfs,
            "search": search_dfs,
            "valid": valid_dfs,
        }

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
