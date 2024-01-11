import itertools
from pathlib import Path
import pickle as pkl
import re
import json
import subprocess

from natsort import natsorted
import pandas as pd


def levin_group_key(pth):
    s = str(pth)
    r = re.search(".*_((Bi)?Levin.*)_e.*(lr0.\d+)", s)
    # r = re.search(".*_((Bi)?Levin)", s)  # ".*(lr0.\d+)", s)
    # r = re.search(".*_((Bi)?Levin).*", s)
    if r:
        return " ".join(r.group(1, 3))
        # return " ".join(r.group(1))


def phs_group_key(pth):
    s = str(pth)
    r = re.search(".*_((Bi)?PHS.*)_e.*(lr0.\d+)", s)
    # r = re.search(".*_((Bi)?Levin)", s)  # ".*(lr0.\d+)", s)
    # r = re.search(".*_((Bi)?Levin).*", s)
    if r:
        return " ".join(r.group(1, 3))
        # return " ".join(r.group(1))


def astar_group_key(pth):
    s = str(pth)
    r = re.search(".*_((Bi)?AStar.*)_e.*(lr0.\d+).*(w\d\.?\d?)", s)
    # r = re.search(".*_((Bi)?AStar).*", s)
    if r:
        return " ".join(r.group(1, 3, 4))

def all_group_key(pth):
    s = str(pth)
    r = re.search(".*_((Bi)?AStar.*)_e.*(lr0.\d+).*(w\d\.?\d?)", s)
    # r = re.search(".*_((Bi)?AStar).*", s)
    if r:
        return " ".join(r.group(1, 3, 4))


def get_runs_data(pth: Path, group_key) -> dict:
    batch_regex = re.compile(".*_b(\d+).pkl")
    runpaths = list(pth.glob("*/"))
    runnames = []
    runpaths = []
    runs = sorted(runpaths, key=group_key)
    for k, g in itertools.groupby(runs, group_key):
        runnames.append(k)
        runpaths.append(list(g))

    runs = {}
    for run_name, run_paths in zip(runnames, runpaths):
        train_dfs = []
        search_dfs = []
        valid_dfs = []
        print(f"Found {len(run_paths)} runs of {run_name}")
        for rp in run_paths:
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
            i += 1
            for j, (tdf, sdf) in enumerate(
                zip(run_train_fs_dfs, run_search_fs_dfs), start=1
            ):
                tdf["stage"] = i
                sdf["stage"] = i
                tdf["epoch"] = j
                sdf["epoch"] = j

            run_valid_df_paths = natsorted(rp.glob("search_valid*.pkl"))
            run_valid_dfs = [pkl.load(p.open("rb")) for p in run_valid_df_paths]

            train_dfs.append(
                pd.concat(run_train_curr_dfs + run_train_fs_dfs, ignore_index=True)
            )
            search_dfs.append(
                pd.concat(run_search_curr_dfs + run_search_fs_dfs, ignore_index=True)
            )
            valid_dfs.append(pd.concat(run_valid_dfs, ignore_index=True))
            validate_batches = [
                int(batch_regex.match(p.name).group(1)) for p in run_valid_df_paths
            ]
            valid_dfs.index = validate_batches

        runs[run_name] = {
            "train": train_dfs,
            "search": search_dfs,
            "valid": valid_dfs,
        }

    return runs


class PdfTemplate:
    def __init__(self, figsdir: Path, outfile: str, toc=True):
        self.figs = figsdir
        self.toc = toc
        self.outfile = outfile
        self.latex_cmds: list = []
        self.text: str = ""

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
