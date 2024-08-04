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
            color=c,
            hatch=hatch,
            alpha=0.1,
        )


def plot_vs_batch(
    domain: str,
    runs_list,
    axs,
    style,
    label=None,
    batch_size=32,
    window_size=200,
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
            color=c,
            hatch=hatch,
            alpha=0.1,
        )
        # For the minor ticks, use no labels; default NullFormatter.


def adaptive_window_size(data, min_periods=250, max_window=None):
    n = len(data)

    # Start with rule of thumb
    window = max(int(n / 15), min_periods)

    # Adjust based on data variability
    std = data.std()
    mean = data.mean()
    cv = std / mean  # Coefficient of variation

    if cv > 0.5:
        window = min(window * 2, n // 3)  # Increase window for high variability
    elif cv < 0.2:
        window = max(window // 2, min_periods)  # Decrease window for low variability

    if max_window:
        window = min(window, max_window)

    return window


def plot_vs_exp(
    domain: str,
    runs_list,
    axs,
    style,
    label=None,
    batch_size=32,
    window_size=250,
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

    central = train_epochs_df["solved"].mean(axis=1)
    central = central.rolling(window=500, min_periods=500).mean()
    x = train_epochs_df["cum_exp"].mean(axis=1)
    axs.plot(x, central, color=c, linestyle=ls, label=label)


def plot_single_domain(domain: str, agents, dom_data: dict, outdir: str):
    saveroot = Path(outdir)
    saveroot.mkdir(exist_ok=True, parents=True)
    # epoch plots
    fig = plt.figure(figsize=(8, 11), dpi=300)
    # plt.close(fig)
    gs = fig.add_gridspec(3, 2)
    bsax = fig.add_subplot(gs[0, 0])
    beax = fig.add_subplot(gs[1, 0])
    vsax = fig.add_subplot(gs[0, 1])
    veax = fig.add_subplot(gs[1, 1])
    eax = fig.add_subplot(gs[2, :])
    # fig.subplots_adjust(hspace=1)

    for agent, runs in dom_data.items():
        if agent not in agents:
            continue

        styles = putils.SingleDomainStyles()
        c, ls, hatch, m = styles.get_ls(agent, True)

        # batch plots
        train_dfs = [r["train"] for r in runs]
        train_batch_df = []
        for df in train_dfs:
            df["solved"] = np.where(df["len"].notna(), True, False)
            dfg = df[["solved", "exp"]]
            dfg = dfg.groupby(df.index // 32)
            dfg = dfg.aggregate({"exp": "mean", "solved": "mean"})
            train_batch_df.append(dfg)

        train_batch_df = pd.concat(train_batch_df, axis=1)

        for col, ax in (
            ("solved", bsax),
            ("exp", beax),
        ):
            # add epoch lines
            for i in range(1, 11):
                x = i * 1562.5
                ax.axvline(x=x, color="k", linestyle=(0, (0.25, 7.5)), linewidth=0.5)

            df = train_batch_df[col]
            smoothed = df.rolling(window=500, min_periods=500).mean()
            # print(smoothed)
            central = smoothed.median(axis=1)
            upper = smoothed.max(axis=1)
            lower = smoothed.min(axis=1)
            # lower = df.min(axis=1)
            # upper = df.max(axis=1)
            # xlabels = df.index.values + 1
            x = np.arange(1, len(df) + 1)
            ax.plot(x, central, color=c, linestyle=ls, label=agent)
            ax.fill_between(
                x,
                lower,
                upper,
                color=c,
                hatch=hatch,
                alpha=0.1,
            )

        # valid plots
        valid_dfs = [r["valid"] for r in runs]
        valid_epochs_df = []
        for df in valid_dfs:
            df["solved"] = np.where(df["len"].notna(), True, False)
            dfg = df[["epoch", "solved", "exp"]]
            dfg = dfg.groupby(df.index // 32)
            dfg = dfg.aggregate({"epoch": "max", "exp": "mean", "solved": "mean"})
            dfg = dfg.groupby(["epoch"], as_index=True).mean()
            valid_epochs_df.append(dfg)

        valid_epochs_df = pd.concat(valid_epochs_df, axis=1)

        for col, ax in (
            ("solved", vsax),
            ("exp", veax),
        ):
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
                label=agent,
            )
            ax.fill_between(
                xticks,
                lower,
                upper,
                color=c,
                hatch=hatch,
                alpha=0.1,
            )

        # plot exp
        train_dfs = [r["train"] for r in runs]
        train_epochs_df = []
        for df in train_dfs:
            df["solved"] = np.where(df["len"].notna(), True, False)
            dfg = df[["solved", "exp"]]
            dfg = dfg.groupby(df.index // 32)
            dfg = dfg.aggregate({"exp": "sum", "solved": "mean"})
            dfg["cum_exp"] = dfg["exp"].cumsum()
            train_epochs_df.append(dfg)

        train_epochs_df = pd.concat(train_epochs_df, axis=1)

        central = train_epochs_df["solved"].mean(axis=1)
        central = central.rolling(window=500, min_periods=500).mean()
        x = train_epochs_df["cum_exp"].mean(axis=1)
        eax.plot(x, central, color=c, linestyle=ls, label=agent)

        # ax1[0].set_ylim(putils.y_lims[domain])

        # make train batch plots
        # for row in range(2):
        #     ax1[row].tick_params(
        #         axis="both",
        #         which="both",
        #         labelsize=16,
        #         width=1.5,
        #         length=4,
        #         direction="inout",
        #     )
        #     ax1[row].tick_params(axis="both", which="major", length=7)
        #     ax = ax1[row]
        #     ax.spines[["right", "top"]].set_visible(False)
        #     ax.xaxis.set_major_locator(MultipleLocator(2))
        #     ax.xaxis.set_major_formatter("{x:.0f}")
        #     ax.xaxis.set_minor_locator(MultipleLocator(1))
        #     if row == 0:
        #         ax.yaxis.set_major_locator(MultipleLocator(0.2))
        #         ax.yaxis.set_major_formatter("{x:.1f}")
        #         ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        #     elif row == 1:
        #         ax.yaxis.set_major_locator(MultipleLocator(1000))
        #         ax.yaxis.set_major_formatter("{x:.0f}")
        #         ax.yaxis.set_minor_locator(MultipleLocator(500))

        # batch plots
        # fig2, ax2 = plt.subplots(2, 1, figsize=(12, 10), dpi=300)
        # plt.close(fig2)
        # ax2[1].set_ylim(putils.y_lims[domain])
        # for row in range(2):
        #     ax2[row].tick_params(
        #         axis="both",
        #         which="both",
        #         labelsize=16,
        #         width=1.5,
        #         length=4,
        #         direction="inout",
        #     )
        #     ax2[row].tick_params(axis="both", which="major", length=7)
        #     ax = ax2[row]
        #     ax.spines[["right", "top"]].set_visible(False)
        #     ax.xaxis.set_major_locator(MultipleLocator(2500))
        #     ax.xaxis.set_major_formatter("{x:.0f}")
        #     ax.xaxis.set_minor_locator(MultipleLocator(500))
        #
        #     if row == 0:
        #         ax.yaxis.set_major_locator(MultipleLocator(0.2))
        #         ax.yaxis.set_major_formatter("{x:0.1f}")
        #         ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        #     elif row == 1:
        #         ax.yaxis.set_major_locator(MultipleLocator(1000))
        #         ax.yaxis.set_major_formatter("{x:.0f}")
        #         ax.yaxis.set_minor_locator(MultipleLocator(500))
        #
        #     # add epoch lines
        #     for i in range(1, 11):
        #         x = i * 1562.5
        #         ax.axvline(x=x, color="k", linestyle=(0, (1, 10)), linewidth=1)
        # # ax[0, 0].set_title(f"Train")
        # # ax[0, 1].set_title(f"Valid")
        # # ax[0, 0].set_ylabel("Solved", size=14)
        # # ax[1, 0].set_ylabel("Expanded", size=14)
        # # ax[2, 0].set_ylabel("Solution length", size=14)
        #
        # fig3, ax3 = plt.subplots(1, 1, figsize=(12, 5), dpi=300)
        # plt.close(fig3)
        # ax3.spines[["right", "top"]].set_visible(False)
        # # ax3.set_ylim(putils.y_lims[domain])
        # ax3.tick_params(
        #     axis="both",
        #     which="both",
        #     labelsize=16,
        #     width=1.5,
        #     length=4,
        #     direction="inout",
        # )
        # ax3.tick_params(axis="both", which="major", length=7)
        # ax3.yaxis.set_major_locator(MultipleLocator(0.2))
        # ax3.yaxis.set_major_formatter("{x:0.1f}")
        # ax3.yaxis.set_minor_locator(MultipleLocator(0.1))
        #
        # # ax3.xaxis.set_major_locator(MultipleLocator(1e9))
        # # ax3.xaxis.set_major_formatter("{x:.0f, 'style':'sci', 'scilimits':(0,0)}")
        # # ax3.xaxis.set_minor_locator(MultipleLocator(2.5e8))
        # ax3.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        #
        # for agent, runs in dom_data.items():
        #     if agent not in agents:
        #         # print(f"Skipping {domain} {agent}")
        #         continue
        #     style = styles.get_ls(agent, True)
        #     plot_vs_epoch(
        #         domain,
        #         runs,
        #         axs=ax1,
        #         style=style,
        #         label=agent,
        #     )
        #     # uncomment for main plots
        #     # style = styles.get_ls(agent, False)
        #     plot_vs_batch(
        #         domain,
        #         runs,
        #         axs=ax2,
        #         style=style,
        #         label=agent,
        #     )
        #
        #     # style = styles.get_ls(agent, True)
        #     plot_vs_exp(
        #         domain,
        #         runs,
        #         axs=ax3,
        #         style=style,
        #         label=agent,
        #     )
        #
        #     print(f"Plotted {domain} {agent}")
        #
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0.5, 1),
        loc="lower center",
        ncols=6,
    )
    fig.savefig(saveroot / f"{domain}.png", bbox_inches="tight")
    plt.close(fig)


def plot_all_domains_batch(group_name, group_agents, doms_path: Path, outdir: str):
    saveroot = Path(outdir)
    saveroot.mkdir(exist_ok=True, parents=True)
    # epoch plots
    sfigs, saxs = plt.subplots(4, 2, sharey=False, figsize=(8, 11), dpi=300)
    plt.close(sfigs)
    efigs, eaxs = plt.subplots(4, 2, sharey=False, figsize=(8, 11), dpi=300)
    plt.close(efigs)
    all_doms = [
        ["tri4", "tri5"],
        ["col4", "col5"],
        ["stp4", "stp5"],
        ["pancake10", "pancake12"],
    ]
    for column, axs in [("solved", saxs), ("exp", eaxs)]:
        for row in range(4):
            for col in range(2):
                ax = axs[row, col]
                dom = all_doms[row][col]
                # ax.set_ylim(putils.y_lims[dom])
                # ax.set_ylim((0, 1.0))
                ax.spines[["right", "top"]].set_visible(False)
                ax.tick_params(
                    axis="both",
                    which="both",
                    labelsize=12,
                    width=1,
                    length=3,
                    direction="inout",
                )
                ax.tick_params(axis="both", which="major", length=5)
                # ax.yaxis.set_major_locator(MultipleLocator(0.2))
                # ax.yaxis.set_major_formatter("{x:0.1f}")
                # ax.yaxis.set_minor_locator(MultipleLocator(0.1))

                # ax3.xaxis.set_major_locator(MultipleLocator(1e9))
                # ax3.xaxis.set_major_formatter("{x:.0f, 'style':'sci', 'scilimits':(0,0)}")
                # ax3.xaxis.set_minor_locator(MultipleLocator(2.5e8))
                ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

                dom_data = pkl.load((doms_path / f"{dom}_trim.pkl").open("rb"))

                styles = putils.AllDomainStyles()
                for agent, runs in dom_data.items():
                    if agent not in group_agents:
                        # print(f"Skipping {domain} {agent}")
                        continue
                    style = styles.get_ls(agent, True)
                    c, ls, hatch, m = style

                    train_dfs = [r["train"] for r in runs]
                    train_batch_dfs = []
                    for df in train_dfs:
                        df["solved"] = np.where(df["len"].notna(), True, False)
                        dfg = df[["solved", "exp"]]
                        dfg = dfg.groupby(df.index // 32)
                        dfg = dfg.aggregate({"exp": "mean", "solved": "mean"})
                        train_batch_dfs.append(dfg)

                    train_batch_dfs = pd.concat(train_batch_dfs, axis=1)

                    df = train_batch_dfs[column]
                    smoothed = df.rolling(window=500, min_periods=500).mean()
                    # print(smoothed)
                    central = smoothed.median(axis=1)
                    upper = smoothed.max(axis=1)
                    lower = smoothed.min(axis=1)
                    # lower = df.min(axis=1)
                    # upper = df.max(axis=1)
                    # xlabels = df.index.values + 1
                    x = np.arange(1, len(df) + 1)
                    ax.plot(x, central, color=c, linestyle=ls, label=agent)
                    ax.fill_between(
                        x,
                        lower,
                        upper,
                        color=c,
                        hatch=hatch,
                        alpha=0.1,
                    )

                    # print(f"Plotted {dom} {agent}")

    sfigs.tight_layout()
    handles, labels = sfigs.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    # figs.legend(by_label.values(), by_label.keys())
    sfigs.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0.5, 1),
        loc="lower center",
        ncols=4,
    )
    sfigs.savefig(saveroot / f"{group_name}_batch_solve.png", bbox_inches="tight")

    efigs.tight_layout()
    handles, labels = efigs.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    # figs.legend(by_label.values(), by_label.keys())
    efigs.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0.5, 1),
        loc="lower center",
        ncols=4,
    )
    efigs.savefig(saveroot / f"{group_name}_batch_exp.png", bbox_inches="tight")


def plot_all_domains_exp(group_name, group_agents, doms_path: Path, outdir: str):
    saveroot = Path(outdir)
    saveroot.mkdir(exist_ok=True, parents=True)
    # epoch plots
    figs, axs = plt.subplots(4, 2, sharey=False, figsize=(8, 11), dpi=300)
    plt.close(figs)
    all_doms = [
        ["tri4", "tri5"],
        ["col4", "col5"],
        ["stp4", "stp5"],
        ["pancake10", "pancake12"],
    ]
    for row in range(4):
        for col in range(2):
            ax = axs[row, col]
            dom = all_doms[row][col]
            # ax.set_ylim(putils.y_lims[dom])
            # ax.set_ylim((0, 1.0))
            ax.spines[["right", "top"]].set_visible(False)
            ax.tick_params(
                axis="both",
                which="both",
                labelsize=12,
                width=1,
                length=3,
                direction="inout",
            )
            ax.tick_params(axis="both", which="major", length=5)
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

            dom_data = pkl.load((doms_path / f"{dom}_trim.pkl").open("rb"))

            styles = putils.AllDomainStyles()
            for agent, runs in dom_data.items():
                if agent not in group_agents:
                    # print(f"Skipping {domain} {agent}")
                    continue
                style = styles.get_ls(agent, True)
                c, ls, hatch, m = style

                train_dfs = [r["train"] for r in runs]
                train_epochs_df = []
                for df in train_dfs:
                    df["solved"] = np.where(df["len"].notna(), True, False)
                    dfg = df[["solved", "exp"]]
                    dfg = dfg.groupby(df.index // 32)
                    dfg = dfg.aggregate({"exp": "sum", "solved": "mean"})
                    dfg["cum_exp"] = dfg["exp"].cumsum()
                    train_epochs_df.append(dfg)

                train_epochs_df = pd.concat(train_epochs_df, axis=1)

                central = train_epochs_df["solved"].mean(axis=1)
                central = central.rolling(window=500, min_periods=500).mean()
                x = train_epochs_df["cum_exp"].mean(axis=1)
                ax.plot(x, central, color=c, linestyle=ls, label=agent)

                # print(f"Plotted {dom} {agent}")

    figs.tight_layout()
    handles, labels = figs.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    # figs.legend(by_label.values(), by_label.keys())
    figs.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0.5, 1),
        loc="lower center",
        ncols=4,
    )
    figs.savefig(saveroot / f"{group_name}_exp.png", bbox_inches="tight")


def main():
    doms_root_path = Path("/home/ken/Projects/thes_data/")
    dom_paths = list(doms_root_path.glob("*trim.pkl"))
    save_dir = "figs/thes_final3/"
    # print(f"{dom.stem} {agent}: {len(runs)} runs")
    # main plots
    # print("Plotting main agents")
    # styles = putils.MixedStyles()
    # for dom in dom_paths:
    #     if dom.stem not in putils.allowable_domains:
    #         continue

    #     agents = dom_agents[dom.stem]
    #     print(f"Plotting {dom.stem}")
    #     dom_data = pkl.load(dom.open("rb"))
    #     plot_domain(dom.stem, agents, dom_data, save_dir, styles)

    # agent_groups = {
    #     "astar": putils.astar_agents,
    # }

    # for gname, gagents in putils.agent_groups.items():
    #     print(f"Plotting {gname} agents")
    #     gname_dir = f"{save_dir}/{gname}"
    #     plot_all_domains_exp(gname, gagents, doms_root_path, gname_dir)
    #     plot_all_domains_batch(gname, gagents, doms_root_path, gname_dir)
    #     print()

    for dom in dom_paths:
        d = dom.stem.split("_")[0]
        gagents = putils.dom_agents[d]
        if d not in putils.allowable_domains:
            continue
        print(f"Plotting {dom.stem}")
        dom_data = pkl.load(dom.open("rb"))
        plot_single_domain(d, gagents, dom_data, f"{save_dir}")
    print()


if __name__ == "__main__":
    main()
