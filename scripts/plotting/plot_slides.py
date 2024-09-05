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

def reorder_agents(dom_data, agent_group=None):
    if agent_group is None:
        all = list(dom_data.items())
    all = [(a, d) for a, d in dom_data.items() if a in agent_group]
    all = sorted(all, key=lambda x: len(x[0]))
    all = sorted(all, key=lambda x: x[0].split("Bi")[-1])
    return OrderedDict(all)

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





def plot_all_domains_:alid(group_name, group_agents, doms_path: Path, outdir: str,filter_agents=None):
    saveroot = Path(outdir)
    saveroot.mkdir(exist_ok=True, parents=True)
    # epoch plots
    sfigs, saxs = plt.subplots(2, 4, sharey=False, figsize=(13.2, 7.2), dpi=300)
    plt.close(sfigs)
    efigs, eaxs = plt.subplots(2, 4, sharey=False, figsize=(13.2, 7.2), dpi=300)
    plt.close(efigs)
    all_doms = [
        ["tri4", "col4", "stp4", "pancake10"],
        ["tri5", "col5", "stp5", "pancake12"],
    ]
    for column, axs in [("solved", saxs), ("exp", eaxs)]:
        for row in range(2):
            for col in range(4):
                ax = axs[row, col]
                dom = all_doms[row][col]
                if isinstance(group_agents, dict):
                    agents = group_agents[dom]
                else:
                    agents = group_agents
                # ax.set_ylim(putils.y_lims[dom])
                # ax.set_ylim((0, 1.0))
                ax.spines[["right", "top"]].set_visible(False)
                # ax.tick_params(
                #     axis="both",
                #     which="both",
                #     labelsize=12,
                #     width=1,
                #     length=3,
                #     direction="inout",
                # )
                # ax.tick_params(axis="both", which="major", length=5)
                # ax.yaxis.set_major_locator(MultipleLocator(0.2))
                # ax.yaxis.set_major_formatter("{x:0.1f}")
                # ax.yaxis.set_minor_locator(MultipleLocator(0.1))

                # ax3.xaxis.set_major_locator(MultipleLocator(1e9))
                # ax3.xaxis.set_major_formatter("{x:.0f, 'style':'sci', 'scilimits':(0,0)}")
                # ax3.xaxis.set_minor_locator(MultipleLocator(2.5e8))
                ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

                dom_data = pkl.load((doms_path / f"{dom}_trim.pkl").open("rb"))
                dom_data = reorder_agents(dom_data, agents)

                styles = putils.SingleDomainStyles()
                for agent, runs in dom_data.items():
                    if agent not in agents:
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
        ncols=6,
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
        ncols=6,
    )
    efigs.savefig(saveroot / f"{group_name}_batch_exp.png", bbox_inches="tight")


def plot_all_domains_batch(group_name, group_agents, doms_path: Path, outdir: str,filter_agents=None):
    saveroot = Path(outdir)
    saveroot.mkdir(exist_ok=True, parents=True)
    # epoch plots
    sfigs, saxs = plt.subplots(2, 4, sharey=False, figsize=(13.2, 7.2), dpi=300)
    plt.close(sfigs)
    efigs, eaxs = plt.subplots(2, 4, sharey=False, figsize=(13.2, 7.2), dpi=300)
    plt.close(efigs)
    all_doms = [
        ["tri4", "col4", "stp4", "pancake10"],
        ["tri5", "col5", "stp5", "pancake12"],
    ]
    for column, axs in [("solved", saxs), ("exp", eaxs)]:
        for row in range(2):
            for col in range(4):
                ax = axs[row, col]
                dom = all_doms[row][col]
                if isinstance(group_agents, dict):
                    agents = group_agents[dom]
                else:
                    agents = group_agents
                # ax.set_ylim(putils.y_lims[dom])
                # ax.set_ylim((0, 1.0))
                ax.spines[["right", "top"]].set_visible(False)
                # ax.tick_params(
                #     axis="both",
                #     which="both",
                #     labelsize=12,
                #     width=1,
                #     length=3,
                #     direction="inout",
                # )
                # ax.tick_params(axis="both", which="major", length=5)
                # ax.yaxis.set_major_locator(MultipleLocator(0.2))
                # ax.yaxis.set_major_formatter("{x:0.1f}")
                # ax.yaxis.set_minor_locator(MultipleLocator(0.1))

                # ax3.xaxis.set_major_locator(MultipleLocator(1e9))
                # ax3.xaxis.set_major_formatter("{x:.0f, 'style':'sci', 'scilimits':(0,0)}")
                # ax3.xaxis.set_minor_locator(MultipleLocator(2.5e8))
                ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

                dom_data = pkl.load((doms_path / f"{dom}_trim.pkl").open("rb"))
                dom_data = reorder_agents(dom_data, agents)

                styles = putils.SingleDomainStyles()
                for agent, runs in dom_data.items():
                    if agent not in agents:
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
        ncols=6,
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
        ncols=6,
    )
    efigs.savefig(saveroot / f"{group_name}_batch_exp.png", bbox_inches="tight")


def plot_all_domains_exp(
    group_name,
    group_agents,
    doms_path: Path,
    outdir: str,
    filter_agents=None,
):
    saveroot = Path(outdir)
    saveroot.mkdir(exist_ok=True, parents=True)
    # epoch plots
    figs, axs = plt.subplots(2, 4, sharey=False, figsize=(13.2, 7.2), dpi=300)
    plt.close(figs)
    all_doms = [
        ["tri4", "col4", "stp4", "pancake10"],
        ["tri5", "col5", "stp5", "pancake12"],
    ]
    for row in range(2):
        for col in range(4):
            ax = axs[row, col]
            dom = all_doms[row][col]
            if isinstance(group_agents, dict):
                agents = group_agents[dom]
            else:
                agents = group_agents
            # ax.set_ylim(putils.y_lims[dom])
            # ax.set_ylim((0, 1.0))
            ax.spines[["right", "top"]].set_visible(False)
            # ax.tick_params(
            #     axis="both",
            #     which="both",
            #     labelsize=12,
            #     width=1,
            #     length=3,
            #     direction="inout",
            # )
            # ax.tick_params(axis="both", which="major", length=5)
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

            dom_data = pkl.load((doms_path / f"{dom}_trim.pkl").open("rb"))
            dom_data = reorder_agents(dom_data, agents)

            styles = putils.SingleDomainStyles()
            for agent, runs in dom_data.items():
                if agent not in agents:
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

                mins = train_epochs_df["solved"].min(axis=1)
                maxs = train_epochs_df["solved"].max(axis=1)
                lower = mins.rolling(window=500, min_periods=500).mean()
                upper = maxs.rolling(window=500, min_periods=500).mean()

                x = train_epochs_df["cum_exp"].mean(axis=1)
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

    figs.tight_layout()
    handles, labels = figs.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    # figs.legend(by_label.values(), by_label.keys())
    figs.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0.5, 1),
        loc="lower center",
        ncols=6,
    )
    figs.savefig(saveroot / f"{group_name}_exp.png", bbox_inches="tight")


def plot_all_domains_grouped_exp(
    group_name,
    group_agents,
    doms_path: Path,
    outdir: str,
    filter_agents=None,
):
    saveroot = Path(outdir)
    saveroot.mkdir(exist_ok=True, parents=True)
    # epoch plots
    all_doms = {"tri": ["tri4", "tri5"], "col": ["col4", "col5"], "stp": ["stp4", "stp5"],
                "pancake": ["pancake10", "pancake12"]}
    for dom, domsized in all_doms.items():
        figs, axs = plt.subplots(1, 2, sharey=False, figsize=(13.2, 7.2), dpi=300)
        plt.close(figs)
        for col in range(2):
            ax = axs[col]
            if isinstance(group_agents, dict):
                agents = group_agents[domsized[col]]
            else:
                agents = group_agents
            # ax.set_ylim(putils.y_lims[dom])
            # ax.set_ylim((0, 1.0))
            ax.spines[["right", "top"]].set_visible(False)
            # ax.tick_params(
            #     axis="both",
            #     which="both",
            #     labelsize=12,
            #     width=1,
            #     length=3,
            #     direction="inout",
            # )
            # ax.tick_params(axis="both", which="major", length=5)
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

            dom_data = pkl.load((doms_path / f"{domsized[col]}_trim.pkl").open("rb"))
            dom_data = reorder_agents(dom_data, agents)

            styles = putils.SingleDomainStyles()
            for agent, runs in dom_data.items():
                # print(f"{domsized[col]} {agent}")
                if agent not in agents:
                    # print(f"Skipping {domsized[col]} {agent}")
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

                mins = train_epochs_df["solved"].min(axis=1)
                maxs = train_epochs_df["solved"].max(axis=1)
                lower = mins.rolling(window=500, min_periods=500).mean()
                upper = maxs.rolling(window=500, min_periods=500).mean()

                x = train_epochs_df["cum_exp"].mean(axis=1)
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

        figs.tight_layout()
        handles, labels = figs.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        # figs.legend(by_label.values(), by_label.keys())
        figs.legend(
            by_label.values(),
            by_label.keys(),
            bbox_to_anchor=(0.5, 1),
            loc="lower center",
            ncols=6,
        )
        figs.savefig(saveroot / f"{group_name}_{dom}_exp.png", bbox_inches="tight")

def main():
    doms_root_path = Path("/home/ken/Envs/trim/")
    save_dir = "figs/thes_slides/"
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

    gname = "best_agents"
    gagents = putils.dom_agents
    print(f"Plotting best agents: solve vs exp plots")
    plot_all_domains_exp(gname, gagents, doms_root_path, save_dir)
    print(f"Plotting best agents, grouped doms: solve vs exp plots")
    plot_all_domains_grouped_exp(gname, gagents, doms_root_path, save_dir)
    print(f"Plotting best agents: vs batch plots")
    # plot_all_domains_time(gname, gagents, doms_root_path, gname_dir)
    plot_all_domains_batch(gname, gagents, doms_root_path, save_dir)
    print()
    # for gname, gagents in putils.agent_groups:
    #     plot_all_domains_exp(gname, gagents, doms_root_path, save_dir)


    # for dom in dom_paths:
    #     d = dom.stem.split("_")[0]
    #     gagents = putils.dom_agents[d]
    #     if d not in putils.allowable_domains:
    #         continue
    #     print(f"Plotting {dom.stem}")
    #     dom_data = pkl.load(dom.open("rb"))
    # plot_single_domain(d, gagents, dom_data, f"{save_dir}")
    print()


if __name__ == "__main__":
    main()
