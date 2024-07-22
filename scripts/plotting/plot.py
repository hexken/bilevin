from collections import OrderedDict
import itertools
from pathlib import Path
import pickle as pkl
from typing import Optional

from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from natsort import natsorted
import numpy as np
import pandas as pd


col4 = (
    "AStar_w1",
    "BiAStar_w1",
    "Levin_nm",
    "BiLevin_nm",
    "PHS_nm",
    "BiPHS_nm",
)

col5 = (
    "AStar_w1",
    "BiAStar_w1",
    "Levin_nm",
    "BiLevin_nm",
    "PHS_nm",
    "BiPHS_nm",
)
tri4 = (
    "AStar_w2.5",
    "BiAStar_w2.5",
    "Levin_nm",
    "BiLevin_nm",
    "PHS_nm",
    "BiPHS_nm",
)

tri5 = (
    "AStar_w1",
    "BiAStar_w1",
    "Levin_nm",
    "BiLevin_nm",
    "PHS_nm",
    "BiPHS_nm",
)
stp4 = (
    "AStar_w2.5",
    "BiAStar_w2.5",
    "Levin_nm",
    "BiLevin_nm",
    "PHS_nm",
    "BiPHS_nm",
)
stp5 = (
    "AStar_w2.5",
    "BiAStar_w2.5",
    "Levin_nm",
    "BiLevin_nm",
    "PHS_nm",
    "BiPHS_nm",
)
bilevin_agents = (
    "BiLevin_m",
    "BiLevin_nm",
    "BiLevinBFS_nm",
)

biphs_agents = (
    "BiPHS_m",
    "BiPHS_nm",
    "BiPHSBFS_nm",
)

biastar_agents = (
    "BiAStar_w1",
    "BiAStar_w2.5",
    "BiAStarBFS_w1",
    "BiAStarBFS_w2.5",
)

levin_agents = (
    "Levin_m",
    "Levin_nm",
)

phs_agents = (
    "PHS_m",
    "PHS_nm",
)

astar_agents = (
    "AStar_w1",
    "AStar_w2.5",
)
allowable_domains = {"stp4", "stp5", "tri4", "tri5", "col4", "col5"}
# allowable_domains = {"stp4", "tri4", "tri5", "col4", "col5"}
# allowable_domains = {"stp5"}


def plot_vs_epoch(
    domain: str,
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
                xticks,
                central,
                color=c,
                linestyle=ls,
                marker=m,
                markerfacecolor="none",
                label=label,
            )
            ax.fill_between(
                xticks,
                lower,
                upper,
                edgecolor=(c, 0.1),
                hatch=hatch,
                facecolor=(c, 0.1),
            )


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
        central = central.rolling(window=175, min_periods=window_size).mean()
        # lower = df.min(axis=1)
        # upper = df.max(axis=1)
        # xlabels = df.index.values + 1
        epochs = np.arange(1, len(df) + 1)
        ax.plot(epochs, central, color=c, linestyle=ls, label=label)
        # For the minor ticks, use no labels; default NullFormatter.


def plot_domain(domain: str, agents, dom_data: dict, outdir: str, styles):
    saveroot = Path(outdir)
    saveroot.mkdir(exist_ok=True, parents=True)
    # epoch plots
    fig1, ax1 = plt.subplots(2, 2, sharey=False, figsize=(12, 10), dpi=300)
    plt.close(fig1)
    ax1[1, 0].set_ylim(y_lims[domain])
    ax1[1, 1].set_ylim(y_lims[domain])

    # batch plots
    fig2, ax2 = plt.subplots(2, 1, figsize=(12, 10), dpi=300)
    plt.close(fig2)
    ax2[1].set_ylim(y_lims[domain])

    for row, col in itertools.product(range(2), range(2)):
        ax1[row, col].tick_params(
            axis="both",
            which="both",
            labelsize=16,
            width=1.5,
            length=4,
            direction="inout",
        )
        ax1[row, col].tick_params(axis="both", which="major", length=7)
        ax = ax1[row, col]
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_major_formatter("{x:.0f}")
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        if row == 0:
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_major_formatter("{x:.1f}")
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        elif row == 1:
            ax.yaxis.set_major_locator(MultipleLocator(500))
            ax.yaxis.set_major_formatter("{x:.0f}")
            ax.yaxis.set_minor_locator(MultipleLocator(250))

    for row in range(2):
        ax2[row].tick_params(
            axis="both",
            which="both",
            labelsize=16,
            width=1.5,
            length=4,
            direction="inout",
        )
        ax2[row].tick_params(axis="both", which="major", length=7)
        ax = ax2[row]
        ax.xaxis.set_major_locator(MultipleLocator(5000))
        ax.xaxis.set_major_formatter("{x:.0f}")
        ax.xaxis.set_minor_locator(MultipleLocator(1000))

        if row == 0:
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_major_formatter("{x:0.1f}")
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        elif row == 1:
            ax.yaxis.set_major_locator(MultipleLocator(500))
            ax.yaxis.set_major_formatter("{x:.0f}")
            ax.yaxis.set_minor_locator(MultipleLocator(250))
    # ax[0, 0].set_title(f"Train")
    # ax[0, 1].set_title(f"Valid")
    # ax[0, 0].set_ylabel("Solved", size=14)
    # ax[1, 0].set_ylabel("Expanded", size=14)
    # ax[2, 0].set_ylabel("Solution length", size=14)

    for agent, runs in dom_data.items():
        if agent not in agents:
            # print(f"Skipping {domain} {agent}")
            continue
        style = styles.get_ls(agent, True)
        plot_vs_epoch(
            domain,
            runs,
            axs=ax1,
            style=style,
            label=agent,
        )
        # uncomment for main plots
        # style = styles.get_ls(agent, False)
        plot_vs_batch(
            domain,
            runs,
            axs=ax2,
            style=style,
            label=agent,
        )

        print(f"Plotted {domain} {agent}")

    fig1.tight_layout()
    handles, labels = fig1.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig1.legend(by_label.values(), by_label.keys())
    fig1.savefig(saveroot / f"{domain}_epoch.png", bbox_inches="tight")

    fig2.tight_layout()
    handles, labels = fig2.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig2.legend(by_label.values(), by_label.keys())
    fig2.savefig(saveroot / f"{domain}_batch.png", bbox_inches="tight")


def main():
    dom_paths = list(Path("/home/ken/Envs/thes_data/").glob("*.pkl"))
    # main plots
    dom_agents = {
        "tri4": tri4,
        "tri5": tri5,
        "stp4": stp4,
        "stp5": stp5,
        "col4": col4,
        "col5": col5,
    }
    print("Plotting main agents")
    styles = MixedStyles()
    for dom in dom_paths:
        if dom.stem not in allowable_domains:
            continue
        agents = dom_agents[dom.stem]
        print(f"Plotting {dom.stem}")
        dom_data = pkl.load(dom.open("rb"))
        plot_domain(dom.stem, agents, dom_data, f"figs/thes/", styles)

        # agent_groups = {
        #     "astar": astar_agents,
        #     "levin": levin_agents,
        #     "phs": phs_agents,
        #     "bilevin": bilevin_agents,
        #     "biphs": biphs_agents,
        #     "biastar": biastar_agents,
        # }
        # for gname, gagents in agent_groups.items():
        #     print(f"Plotting {gname} agents")
        #     for dom in dom_paths:
        #         styles = SequentialStyles()
        #         if dom.stem not in allowable_domains:
        #             continue
        #         print(f"Plotting {dom.stem}")
        #         dom_data = pkl.load(dom.open("rb"))
        #         plot_domain(dom.stem, gagents, dom_data, f"figs/thes/{gname}", styles)
        print()


class SequentialStyles:
    def __init__(self):
        self.marker = ["o", "x"]
        self.ls = ["-", (0, (5, 6))]
        self.hatch = [None, "|||"]
        self.seq_colors = ["r", "g", "b", "c", "m", "y"]
        # lighter colors earlier in the list
        self.mi = 0
        self.lsi = 0
        self.hi = 0
        self.ci = 0

    def get_ls(self, agent: str, same_color: bool):
        color = self.seq_colors[self.ci]
        ls = self.ls[self.lsi]
        hatch = self.hatch[self.hi]
        m = self.marker[self.mi]

        self.mi = (self.mi + 1) % len(self.marker)
        self.lsi = (self.lsi + 1) % len(self.ls)
        self.hi = (self.hi + 1) % len(self.hatch)
        self.ci = (self.ci + 1) % len(self.seq_colors)

        return color, ls, hatch, m


class MixedStyles:
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
