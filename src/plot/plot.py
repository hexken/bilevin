import itertools
import json
from pathlib import Path
import pickle as pkl
import re
import subprocess
from typing import Optional
import warnings

from cycler import cycler
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import pandas as pd

from utils import PdfTemplate, all_group_key, get_runs_data, ColorMapper

cmap = mpl.colormaps["tab20"]
mpl.rcParams["axes.prop_cycle"] = cycler(color=cmap.colors)
mpl.rcParams["axes.linewidth"] = 1
mpl.rc("lines", linewidth=2, linestyle="-")


def plot_solve_vs_time(runs_data, batch_size=40, window_size=150):
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = []

    cmapper = ColorMapper()

    for run_name, run_data in runs_data.items():
        color = cmapper.get_color(run_name)
        dfs = run_data["search"]
        solveds = []
        cols = []
        for i, df in enumerate(dfs):
            df = df[["time", "len"]].copy()
            df["len"] = df["len"].apply(pd.notna)
            dfg = df.groupby(df.index // batch_size)
            dfs = dfg.aggregate({"time": "max", "len": "mean"})
            dfs["time"] = dfs["time"].cumsum()
            x = dfs.groupby(dfs["time"] // window_size)["len"].mean()
            x.index = x.index.map(lambda x: (x + 1) * window_size)
            solveds.append(x)
            cols.append(i)

        df = pd.concat(solveds, axis=1)
        df.columns = cols
        df["mean"] = df.mean(axis=1)
        df["min"] = df.min(axis=1)
        df["max"] = df.max(axis=1)

        # color = tab_colors.popitem()[1]
        # color = colors.pop()
        ax.plot(df.index.values, df["mean"], label=run_name, color=color)
        plt.fill_between(
            np.array(df.index.values, dtype=np.float32),
            df["min"],
            df["max"],
            alpha=0.1,
            color=color,
            label=run_name,
        )
        labels.append(run_name)

    handler, labeler = ax.get_legend_handles_labels()
    it = iter(handler)
    hd = [(a, next(it)) for a in it]
    ax.legend(hd, labels)
    ax.set_xlabel("Time (s)", size=12)
    ax.set_ylabel("Solved", rotation=0, labelpad=30, size=12)
    ax.set_title("Solved vs. time")
    plt.show()
    # plt.savefig(figs_dir / f"svt_{str(exp_name)}.pdf")


def main():
    all_runs_pth = Path("/home/ken/Projects/bilevin/final_runs_test/").glob("stp4*")
    all_runs = get_runs_data(all_runs_pth, all_group_key)
    plot_solve_vs_time(all_runs, batch_size=40, window_size=150)
    # plt.show()


if __name__ == "__main__":
    main()
