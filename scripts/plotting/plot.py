from collections import OrderedDict
import itertools
import os
from pathlib import Path
import pickle as pkl
import sys
from typing import Optional

from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from natsort import natsorted
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import plotting.utils as putils


# def plot_vs_epoch(
#     domain: str,
#     runs_list,
#     axs,
#     style,
#     label=None,
#     batch_size=32,
# ):
#     # plot seeds corresponding to a run
#     train_dfs = [r["train"] for r in runs_list]
#     train_epochs_df = []
#     for df in train_dfs:
#         df["solved"] = np.where(df["len"].notna(), True, False)
#         dfg = df[["epoch", "solved", "exp"]]
#         # dfg = dfg.groupby(df.index // batch_size)
#         # dfg = dfg.aggregate({"epoch": "max", "exp": "mean", "solved": "mean"})
#         dfg = dfg.groupby(["epoch"], as_index=True).mean()
#         train_epochs_df.append(dfg)

#     train_epochs_df = pd.concat(train_epochs_df, axis=1)

#     valid_dfs = [r["valid"] for r in runs_list]
#     valid_epochs_df = []
#     for df in valid_dfs:
#         df["solved"] = np.where(df["len"].notna(), True, False)
#         dfg = df[["epoch", "solved", "exp"]]
#         dfg = dfg.groupby(df.index // batch_size)
#         dfg = dfg.aggregate({"epoch": "max", "exp": "mean", "solved": "mean"})
#         dfg = dfg.groupby(["epoch"], as_index=True).mean()
#         valid_epochs_df.append(dfg)

#     valid_epochs_df = pd.concat(valid_epochs_df, axis=1)

#     c, ls, hatch, m = style

#     for col, axr in (
#         ("solved", 0),
#         ("exp", 1),
#     ):
#         for dat, axc in ((train_epochs_df, 0), (valid_epochs_df, 1)):
#             ax = axs[axr, axc]
#             df = dat[col]
#             central = df.median(axis=1)
#             lower = df.min(axis=1)
#             upper = df.max(axis=1)
#             xlabels = df.index.values
#             xticks = np.arange(1, len(xlabels) + 1)

#             ax.plot(
#                 xticks,
#                 central,
#                 color=c,
#                 linestyle=ls,
#                 marker=m,
#                 markerfacecolor="none",
#                 label=label,
#             )
#             ax.fill_between(
#                 xticks,
#                 lower,
#                 upper,
#                 edgecolor=(c, 0.1),
#                 hatch=hatch,
#                 facecolor=(c, 0.1),
# )


def plot_vs_epoch(
    domain: str,
    runs_list,
    axs,
    style,
    label=None,
    batch_size=32,
):
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
        ax = axs[axr]
        df = valid_epochs_df[col]
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

    for col, axr in (
        ("solved", 0),
        ("exp", 1),
    ):
        ax = axs[axr]
        df = train_epochs_df[col]
        smoothed = df.rolling(window=175, min_periods=window_size).mean()
        # print(smoothed)
        central = smoothed.median(axis=1)
        upper = smoothed.max(axis=1)
        lower = smoothed.min(axis=1)
        # lower = df.min(axis=1)
        # upper = df.max(axis=1)
        # xlabels = df.index.values + 1
        x = np.arange(1, len(df) + 1)
        ax.plot(x, central, color=c, linestyle=ls, label=label)
        ax.fill_between(
            x,
            lower,
            upper,
            edgecolor=(c, 0.1),
            hatch=hatch,
            facecolor=(c, 0.1),
        )
        # For the minor ticks, use no labels; default NullFormatter.


def plot_vs_exp(
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
        dfg = dfg.aggregate({"exp": "sum", "solved": "mean"})
        dfg["cum_exp"] = dfg["exp"].cumsum()
        train_epochs_df.append(dfg)

    train_epochs_df = pd.concat(train_epochs_df, axis=1)

    c, ls, hatch, m = style

    y = train_epochs_df["solved"].mean(axis=1)
    y = y.rolling(window=175, min_periods=window_size).mean()
    x = train_epochs_df["cum_exp"].mean(axis=1)
    # y = train_epochs_df.mean(axis=1)
    # central = central.rolling(window=175, min_periods=window_size).mean()
    # lower = df.min(axis=1)
    # upper = df.max(axis=1)
    # xlabels = df.index.values + 1
    axs.plot(x, y, color=c, linestyle=ls, label=label)


def plot_domain(domain: str, agents, dom_data: dict, outdir: str, styles):
    saveroot = Path(outdir)
    saveroot.mkdir(exist_ok=True, parents=True)
    # epoch plots
    fig1, ax1 = plt.subplots(2, 1, sharey=False, figsize=(12, 10), dpi=300)
    plt.close(fig1)
    # ax1[0].set_ylim(putils.y_lims[domain])
    ax1[1].set_ylim(putils.y_lims[domain])
    for row in range(2):
        ax1[row].tick_params(
            axis="both",
            which="both",
            labelsize=16,
            width=1.5,
            length=4,
            direction="inout",
        )
        ax1[row].tick_params(axis="both", which="major", length=7)
        ax = ax1[row]
        ax.spines[["right", "top"]].set_visible(False)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_major_formatter("{x:.0f}")
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        if row == 0:
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_major_formatter("{x:.1f}")
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        elif row == 1:
            ax.yaxis.set_major_locator(MultipleLocator(1000))
            ax.yaxis.set_major_formatter("{x:.0f}")
            ax.yaxis.set_minor_locator(MultipleLocator(500))

    # batch plots
    fig2, ax2 = plt.subplots(2, 1, figsize=(12, 10), dpi=300)
    plt.close(fig2)
    ax2[1].set_ylim(putils.y_lims[domain])
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
        ax.spines[["right", "top"]].set_visible(False)
        ax.xaxis.set_major_locator(MultipleLocator(2500))
        ax.xaxis.set_major_formatter("{x:.0f}")
        ax.xaxis.set_minor_locator(MultipleLocator(500))

        if row == 0:
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_major_formatter("{x:0.1f}")
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        elif row == 1:
            ax.yaxis.set_major_locator(MultipleLocator(1000))
            ax.yaxis.set_major_formatter("{x:.0f}")
            ax.yaxis.set_minor_locator(MultipleLocator(500))

        # add epoch lines
        for i in range(1, 11):
            x = i * 1562.5
            ax.axvline(x=x, color="k", linestyle=(0, (1, 10)), linewidth=1)
    # ax[0, 0].set_title(f"Train")
    # ax[0, 1].set_title(f"Valid")
    # ax[0, 0].set_ylabel("Solved", size=14)
    # ax[1, 0].set_ylabel("Expanded", size=14)
    # ax[2, 0].set_ylabel("Solution length", size=14)

    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 5), dpi=300)
    plt.close(fig3)
    ax3.spines[["right", "top"]].set_visible(False)
    # ax3.set_ylim(putils.y_lims[domain])
    ax3.tick_params(
        axis="both",
        which="both",
        labelsize=16,
        width=1.5,
        length=4,
        direction="inout",
    )
    ax3.tick_params(axis="both", which="major", length=7)
    ax3.yaxis.set_major_locator(MultipleLocator(0.2))
    ax3.yaxis.set_major_formatter("{x:0.1f}")
    ax3.yaxis.set_minor_locator(MultipleLocator(0.1))

    # ax3.xaxis.set_major_locator(MultipleLocator(1e9))
    # ax3.xaxis.set_major_formatter("{x:.0f, 'style':'sci', 'scilimits':(0,0)}")
    # ax3.xaxis.set_minor_locator(MultipleLocator(2.5e8))
    ax3.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

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

        # style = styles.get_ls(agent, True)
        plot_vs_exp(
            domain,
            runs,
            axs=ax3,
            style=style,
            label=agent,
        )

        print(f"Plotted {domain} {agent}")

    fig1.tight_layout()
    handles, labels = fig1.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig1.legend(by_label.values(), by_label.keys())
    fig1.savefig(saveroot / f"{domain}_valid.png", bbox_inches="tight")

    fig2.tight_layout()
    handles, labels = fig2.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig2.legend(by_label.values(), by_label.keys())
    fig2.savefig(saveroot / f"{domain}_batch.png", bbox_inches="tight")

    fig3.tight_layout()
    handles, labels = fig3.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig3.legend(by_label.values(), by_label.keys())
    fig3.savefig(saveroot / f"{domain}_exp.png", bbox_inches="tight")


def main():
    dom_paths = list(Path("/home/ken/Envs/thes_data/").glob("*.pkl"))
    save_dir = "figs/thes_final/"
    # main plots
    dom_agents = {
        "tri4": putils.tri4,
        "tri5": putils.tri5,
        "stp4": putils.stp4,
        "stp5": putils.stp5,
        "col4": putils.col4,
        "col5": putils.col5,
    }
    print("Plotting main agents")
    styles = putils.MixedStyles()
    for dom in dom_paths:
        if dom.stem not in putils.allowable_domains:
            continue

        agents = dom_agents[dom.stem]
        print(f"Plotting {dom.stem}")
        dom_data = pkl.load(dom.open("rb"))
        plot_domain(dom.stem, agents, dom_data, save_dir, styles)

    agent_groups = {
        "astar": putils.astar_agents,
        "levin": putils.levin_agents,
        "phs": putils.phs_agents,
        "bilevin": putils.bilevin_agents,
        "biphs": putils.biphs_agents,
        "biastar": putils.biastar_agents,
    }
    for gname, gagents in agent_groups.items():
        print(f"Plotting {gname} agents")
        for dom in dom_paths:
            styles = putils.SequentialStyles()
            if dom.stem not in putils.allowable_domains:
                continue
            print(f"Plotting {dom.stem}")
            dom_data = pkl.load(dom.open("rb"))
            plot_domain(dom.stem, gagents, dom_data, f"{save_dir}/{gname}", styles)
        print()


if __name__ == "__main__":
    main()
