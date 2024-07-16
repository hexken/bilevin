from collections import OrderedDict
import itertools
from pathlib import Path
import pickle as pkl
from typing import Optional

from cycler import cycler
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import pandas as pd

cmap = mpl.colormaps["tab20"]
mpl.rcParams["axes.linewidth"] = 1
mpl.rc("lines", linewidth=1, markersize=3)


agent_order = ("PHS", "BiPHS", "Levin", "BiLevin", "AStar", "BiAStar")
allowable_domains = {"stp4", "tri4", "tri5", "col4", "col5"}

main_agents = (
    "AStar_w2.5",
    "BiAStar_w2.5",
    "Levin_nm",
    "BiLevin_nm",
    "PHS_nm",
    "BiPHS_nm",
)
mask_agents = (
    "Levin_m",
    "BiLevin_m",
    "PHS_m",
    "BiPHS_m",
)
weight_agents = (
    "AStar_w2.5",
    "BiAStar_w.5",
    "AStar_w1",
    "BiAStar_w1",
)


def plot_vs_epoch(
    runs_list,
    axs,
    style,
    label=None,
    batch_size=32,
):
    # plot seeds corresponding to a run
    train_dfs = [r["train"] for r in runs_list]
    train_epochs_df = []
    for df in train_dfs:
        df["solved"] = np.where(df["len"].notna(), True, False)
        dfg = df[["epoch", "solved", "exp"]]
        dfg = dfg.groupby(df.index // batch_size)
        dfg = dfg.aggregate({"epoch": "max", "exp": "mean", "solved": "mean"})
        dfg = dfg.groupby(["epoch"], as_index=True).mean()
        train_epochs_df.append(dfg)

    train_epochs_df = pd.concat(train_epochs_df, axis=1)

    valid_dfs = [r["valid"] for r in runs_list]
    valid_epochs_df = []
    for df in valid_dfs:
        df["solved"] = np.where(df["len"].notna(), True, False)
        dfg = df[["epoch", "solved", "exp"]]
        dfg = dfg.groupby(df.index // batch_size)
        dfg = dfg.aggregate({"epoch": "max", "exp": "mean", "solved": "mean"})
        dfg = dfg.groupby(["epoch"], as_index=True).mean()
        valid_epochs_df.append(dfg)

    valid_epochs_df = pd.concat(valid_epochs_df, axis=1)
    c, ls, hatch, m = style
    if c == "b":
        print(valid_epochs_df)

    for col, axt in (
        ("solved", 0),
        ("exp", 1),
    ):
        ax = axs[axt, 0]
        df = train_epochs_df[col]
        central = df.median(axis=1)
        lower = df.min(axis=1)
        upper = df.max(axis=1)
        xlabels = df.index.values
        xticks = np.arange(1, len(xlabels) + 1)

        ax.plot(xticks, central, color=c, linestyle=ls, marker=m)
        ax.fill_between(
            xticks,
            lower,
            upper,
            edgecolor=(c, 0.1),
            hatch=hatch,
            facecolor=(c, 0.1),
            label=label,
        )
        ax.set_xticks(xticks, xlabels, rotation=70)

        ax = axs[axt, 1]
        df = valid_epochs_df[col]
        central = df.median(axis=1)
        lower = df.min(axis=1)
        upper = df.max(axis=1)
        xlabels = df.index.values
        xticks = np.arange(1, len(xlabels) + 1)

        c, ls, hatch, m = style
        ax.plot(xticks, central, color=c, linestyle=ls, marker=m)
        ax.fill_between(
            xticks,
            lower,
            upper,
            edgecolor=(c, 0.1),
            hatch=hatch,
            facecolor=(c, 0.1),
            label=label,
        )
        ax.set_xticks(xticks, xlabels, rotation=70)

def plot_vs_batch(
    runs_list,
    axs,
    style,
    label=None,
    batch_size=32,
    window_size=100,
):
    # plot seeds corresponding to a run
    train_dfs = [r["train"] for r in runs_list]
    train_epochs_df = []
    for df in train_dfs:
        df["solved"] = np.where(df["len"].notna(), True, False)
        dfg = df[["epoch", "solved", "exp"]]
        dfg = dfg.groupby(df.index // batch_size)
        dfg = dfg.aggregate({"epoch": "max", "exp": "mean", "solved": "mean"})
        dfg = dfg.groupby(["epoch"], as_index=True).mean()
        train_epochs_df.append(dfg)

    train_epochs_df = pd.concat(train_epochs_df, axis=1)

    valid_dfs = [r["valid"] for r in runs_list]
    valid_epochs_df = []
    for df in valid_dfs:
        df["solved"] = np.where(df["len"].notna(), True, False)
        dfg = df[["epoch", "solved", "exp"]]
        dfg = dfg.groupby(df.index // batch_size)
        dfg = dfg.aggregate({"epoch": "max", "exp": "mean", "solved": "mean"})
        dfg = dfg.groupby(["epoch"], as_index=True).mean()
        valid_epochs_df.append(dfg)

    valid_epochs_df = pd.concat(valid_epochs_df, axis=1)
    c, ls, hatch, m = style
    if c == "b":
        print(valid_epochs_df)

    for col, axt in (
        ("solved", 0),
        ("exp", 1),
    ):
        ax = axs[axt, 0]
        df = train_epochs_df[col]
        central = df.median(axis=1)
        lower = df.min(axis=1)
        upper = df.max(axis=1)
        xlabels = df.index.values
        xticks = np.arange(1, len(xlabels) + 1)

        ax.plot(xticks, central, color=c, linestyle=ls, marker=m)
        ax.fill_between(
            xticks,
            lower,
            upper,
            edgecolor=(c, 0.1),
            hatch=hatch,
            facecolor=(c, 0.1),
            label=label,
        )
        ax.set_xticks(xticks, xlabels, rotation=70)

        ax = axs[axt, 1]
        df = valid_epochs_df[col]
        central = df.median(axis=1)
        lower = df.min(axis=1)
        upper = df.max(axis=1)
        xlabels = df.index.values
        xticks = np.arange(1, len(xlabels) + 1)

        c, ls, hatch, m = style
        ax.plot(xticks, central, color=c, linestyle=ls, marker=m)
        ax.fill_between(
            xticks,
            lower,
            upper,
            edgecolor=(c, 0.1),
            hatch=hatch,
            facecolor=(c, 0.1),
            label=label,
        )
        ax.set_xticks(xticks, xlabels, rotation=70)

def plot_domain(domain: str, agents, dom_data: dict, outdir: str):
    saveroot = Path(outdir)
    saveroot.mkdir(exist_ok=True, parents=True)
    ls_mapper = LineStyleMapper()
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    # ax[0, 0].set_title(f"Train")
    # ax[0, 1].set_title(f"Valid")
    # ax[0, 0].set_ylabel("Solved", size=14)
    # ax[1, 0].set_ylabel("Expanded", size=14)
    # ax[2, 0].set_ylabel("Solution length", size=14)

    # loop over agents (uni/bi - levin/phs/astar, etc.)
    for agent, runs in dom_data.items():
        if agent not in agents:
            # print(f"Skipping {domain} {agent}")
            continue
        style = ls_mapper.get_ls(agent)
        # fill search and val plots
        plot_vs_epoch(
            runs,
            axs=ax,
            style=style,
            label=agent,
        )

        print(f"Plotted {domain} {agent}")
    fig.tight_layout()
    fig.savefig(saveroot / f"{domain}_train.pdf", bbox_inches="tight")
    plt.close()


def main():
    dom_paths = list(Path("/home/ken/Envs/thes_good/").glob("*.pkl"))
    for dom in dom_paths:
        if dom.stem not in allowable_domains:
            continue
        print(f"Plotting {dom.stem}")
        dom_data = pkl.load(dom.open("rb"))
        # new_dom_data = OrderedDict()
        # for order_agent in order:
        #     for agent in dom_data:
        #         base_agent = agent.split()[0]
        #         if base_agent == order_agent:
        #             new_dom_data[agent] = dom_data[agent]
        plot_domain(dom.stem, main_agents, dom_data, f"figs/july/")


class LineStyleMapper:
    def __init__(self):
        self.uni_marker = "o"
        self.bi_marker = "x"
        self.uni_ls = "-"
        self.bi_lds = "--"
        self.bibfs_ls = "--"
        self.bialt_ls = ":"
        self.bi_hatch = "||"
        self.uni_hatch = None

    def get_ls(self, s: str):
        if "AStar" in s:
            c = "r"
        elif "Levin" in s:
            c = "g"
        elif "PHS" in s:
            c = "b"
        else:
            raise ValueError(f"Unknown algorithm: {s}")

        # if "Alt" in s:
        #     ls = self.bialt_ls
        # elif "BFS" in s:
        #     ls = self.bibfs_ls
        # else:
        #     ls = self.uni_ls

        if "Bi" in s:
            ls = self.bibfs_ls
            h = self.bi_hatch
            m = self.bi_marker
        else:
            ls = self.uni_ls
            h = self.uni_hatch
            m = self.uni_marker

        return c, ls, h, m


if __name__ == "__main__":
    main()
