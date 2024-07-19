from collections import OrderedDict
import itertools
from pathlib import Path
import pickle as pkl
from typing import Optional
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

from cycler import cycler
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import pandas as pd

cmap = mpl.colormaps["tab20"]
mpl.rcParams["axes.linewidth"] = 1
mpl.rc("lines", linewidth=1, markersize=7)


agent_order = ("PHS", "BiPHS", "Levin", "BiLevin", "AStar", "BiAStar")
allowable_domains = {"stp4", "stp5", "tri4", "tri5", "col4", "col5"}
# allowable_domains = {"stp4", "tri4", "tri5", "col4", "col5"}
# allowable_domains = {"stp5"}

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
    domain: str,
    runs_list,
    axs,
    style,
    batch_size=32,
):
    # plot seeds corresponding to a run
    train_dfs = [r["train"] for r in runs_list]
    train_epochs_df = []
    for df in train_dfs:
        df["solved"] = np.where(df["len"].notna(), True, False)
        dfg = df[["epoch", "solved", "exp"]]
        # dfg = dfg.groupby(df.index // batch_size)
        # dfg = dfg.aggregate({"epoch": "max", "exp": "mean", "solved": "mean"})
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

    for col, axr in (
        ("solved", 0),
        ("exp", 1),
    ):
        for dat, axc in ((train_epochs_df, 0), (valid_epochs_df, 1)):
            ax = axs[axr, axc]
            df = dat[col]
            central = df.median(axis=1)
            lower = df.min(axis=1)
            upper = df.max(axis=1)
            xlabels = df.index.values
            xticks = np.arange(1, len(xlabels) + 1)

            ax.plot(
                xticks, central, color=c, linestyle=ls, marker=m, markerfacecolor="none"
            )
            ax.fill_between(
                xticks,
                lower,
                upper,
                edgecolor=(c, 0.1),
                hatch=hatch,
                facecolor=(c, 0.1),
            )
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_major_formatter("{x:.0f}")
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            if axr == 0:
                ax.yaxis.set_major_locator(MultipleLocator(0.2))
                ax.yaxis.set_major_formatter("{x:.1f}")
                ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            elif axr == 1:
                ax.yaxis.set_major_locator(MultipleLocator(500))
                ax.yaxis.set_major_formatter("{x:.0f}")
                ax.yaxis.set_minor_locator(MultipleLocator(250))


y_lims = {
    "tri4": (0, 1900),
    "tri5": (0, 4000),
    "stp4": (500, 4100),
    "stp5": (1000, 7250),
    "col4": (0, 1800),
    "col5": (0, 4100),
}


def plot_vs_batch(
    domain: str,
    runs_list,
    axs,
    style,
    label=None,
    batch_size=32,
    window_size=175,
):
    train_dfs = [r["train"] for r in runs_list]
    train_epochs_df = []
    for df in train_dfs:
        df["solved"] = np.where(df["len"].notna(), True, False)
        dfg = df[["solved", "exp"]]
        dfg = dfg.groupby(df.index // batch_size)
        dfg = dfg.aggregate({"exp": "mean", "solved": "mean"})
        train_epochs_df.append(dfg)

    train_epochs_df = pd.concat(train_epochs_df, axis=1)

    c, ls, hatch, m = style

    for col, axt in (
        ("solved", 0),
        ("exp", 1),
    ):
        ax = axs[axt]
        df = train_epochs_df[col]
        central = df.mean(axis=1)
        central = central.ewm(alpha=0.25).mean()
        # lower = df.min(axis=1)
        # upper = df.max(axis=1)
        # xlabels = df.index.values + 1
        epochs = np.arange(1, len(df) + 1)
        ax.plot(epochs, central, color=c, linestyle=ls)
        ax.xaxis.set_major_locator(MultipleLocator(5000))
        ax.xaxis.set_major_formatter("{x:.0f}")
        ax.xaxis.set_minor_locator(MultipleLocator(1000))

        if axt == 0:
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_major_formatter("{x:0.1f}")
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        elif axt == 1:
            ax.yaxis.set_major_locator(MultipleLocator(500))
            ax.yaxis.set_major_formatter("{x:.0f}")
            ax.yaxis.set_minor_locator(MultipleLocator(250))
        # For the minor ticks, use no labels; default NullFormatter.


def plot_domain(domain: str, agents, dom_data: dict, outdir: str):
    saveroot = Path(outdir)
    saveroot.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(12, 10), dpi=300)
    fig2, ax2 = plt.subplots(2, 1, figsize=(12, 10), dpi=300)
    ax[1, 0].set_ylim(y_lims[domain])
    ax[1, 1].set_ylim(y_lims[domain])
    ax2[1].set_ylim(y_lims[domain])
    for i, j in itertools.product(range(2), range(2)):
        ax[i, j].tick_params(
            axis="both",
            which="both",
            labelsize=16,
            width=1.5,
            length=4,
            direction="inout",
        )
        ax[i, j].tick_params(axis="both", which="major", length=7)
    for i in range(2):
        ax2[i].tick_params(
            axis="both",
            which="both",
            labelsize=16,
            width=1.5,
            length=4,
            direction="inout",
        )
        ax2[i].tick_params(axis="both", which="major", length=7)
    # ax[0, 0].set_title(f"Train")
    # ax[0, 1].set_title(f"Valid")
    # ax[0, 0].set_ylabel("Solved", size=14)
    # ax[1, 0].set_ylabel("Expanded", size=14)
    # ax[2, 0].set_ylabel("Solution length", size=14)

    ls_mapper = LineStyleMapper()

    for agent, runs in dom_data.items():
        if agent not in agents:
            # print(f"Skipping {domain} {agent}")
            continue
        style = ls_mapper.get_ls(agent, True)
        plot_vs_epoch(
            domain,
            runs,
            axs=ax,
            style=style,
        )
        style = ls_mapper.get_ls(agent, False)
        plot_vs_batch(
            domain,
            runs,
            axs=ax2,
            style=style,
        )

        print(f"Plotted {domain} {agent}")

    fig.tight_layout()
    fig.savefig(saveroot / f"{domain}_epoch.png", bbox_inches="tight")
    fig2.tight_layout()
    fig2.savefig(saveroot / f"{domain}_batch.png", bbox_inches="tight")
    plt.close()


def main():
    dom_paths = list(Path("/home/ken/Envs/thestest2/").glob("*.pkl"))
    for dom in dom_paths:
        if dom.stem not in allowable_domains:
            continue
        print(f"Plotting {dom.stem}")
        dom_data = pkl.load(dom.open("rb"))
        plot_domain(dom.stem, main_agents, dom_data, f"figs/july2/")


class LineStyleMapper:
    def __init__(self):
        self.uni_marker = "o"
        self.bi_marker = "x"
        self.uni_ls = "-"
        self.bi_lds = "--"
        self.bibfs_ls = (0, (5, 6))
        self.bialt_ls = ":"
        self.bi_hatch = "|||"
        self.uni_hatch = None
        self.colors = ["#FF0000", "#900000", "#00FF00", "#009000", "#0AA0F5", "#000070"]
        # lighter colors earlier in the list

    def get_ls(self, agent: str, same_color: bool):
        s = agent.split("_")[0]
        if s == "AStar":
            ci = 1
        elif s == "BiAStar":
            ci = 0
        elif s == "Levin":
            ci = 3
        elif s == "BiLevin":
            ci = 2
        elif s == "PHS":
            ci = 5
        elif s == "BiPHS":
            ci = 4
        else:
            raise ValueError(f"Invalid agent {s}")

        # if "Alt" in s:
        #     ls = self.bialt_ls
        # elif "BFS" in s:
        #     ls = self.bibfs_ls
        # else:
        #     ls = self.uni_ls

        if "Bi" in s:
            h = self.bi_hatch
            m = self.bi_marker
        else:
            h = self.uni_hatch
            m = self.uni_marker

        ls = self.uni_ls

        if same_color:
            if "Bi" in s:
                ls = self.bibfs_ls
                ci += 1

        return self.colors[ci], ls, h, m


if __name__ == "__main__":
    main()
