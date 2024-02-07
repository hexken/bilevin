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

import plotting.keys as pkeys
from plotting.utils import (
    LineStyleMapper,
    PdfTemplate,
    get_runs_data,
)

cmap = mpl.colormaps["tab20"]
mpl.rcParams["axes.prop_cycle"] = cycler(color=cmap.colors)
mpl.rcParams["axes.linewidth"] = 1
mpl.rc("lines", linewidth=1, linestyle="-")


def plot_search_vs_time(
    run_data,
    y_data_label: str,
    ax,
    color=None,
    linestyle=None,
    label=None,
    batch_size=4,
    window_size=50,
):
    # plot seeds corresponding to a run
    dfs = run_data["search"]
    notna_dfs = []
    for df in dfs:
        dfg = df[["time", y_data_label]].copy()
        dfg = dfg.groupby(df.index // batch_size)
        dfg = dfg.aggregate({"time": "max", y_data_label: "mean"})
        dfg["time"] = dfg["time"].cumsum()
        dfg = dfg.groupby(dfg["time"] // window_size)[y_data_label].mean()
        dfg.index = dfg.index.map(lambda x: (x + 1) * window_size)
        notna_dfs.append(dfg)

    df = pd.concat(notna_dfs, axis=1)
    means = df.mean(axis=1)
    mins = df.min(axis=1)
    maxs = df.max(axis=1)

    ax.plot(df.index.values, means, color=color, linestyle=linestyle, label=label)
    ax.fill_between(
        np.array(df.index.values, dtype=np.float32),
        mins,
        maxs,
        alpha=0.1,
        color=color,
        label=label,
    )


def plot_valid_vs_time(
    run_data,
    y_data_label: str,
    ax,
    color=None,
    linestyle=None,
    label=None,
):
    # plot seeds corresponding to a run
    dfs = run_data["valid"]
    notna_dfs = []
    xs = []
    for df in dfs:
        x = df["start_time"].unique()
        xs.append(x)

        dfg = df.groupby(df["batch"], as_index=False).mean()[y_data_label]
        notna_dfs.append(dfg)

    min_len = min(len(x) for x in xs)
    notna_dfs = [df.iloc[:min_len] for df in notna_dfs]
    df = pd.concat(notna_dfs, axis=1)
    xs = [x[:min_len] for x in xs]
    xs = np.array(xs).mean(axis=0)
    means = df.mean(axis=1)
    mins = df.min(axis=1)
    maxs = df.max(axis=1)

    ax.plot(xs, means, color=color, linestyle=linestyle, label=label)
    ax.fill_between(
        xs,
        mins,
        maxs,
        alpha=0.1,
        color=color,
        label=label,
    )


def window_avg_over_index(data: list[pd.DataFrame], y_data_label, window_size=5):
    aggr_dfs = []
    for df in data:
        df = df[y_data_label]
        df = df.groupby(df.index // window_size).mean()
        aggr_dfs.append(df)

    df = pd.concat(aggr_dfs, axis=1)
    xs = df.index.map(lambda x: x // window_size)
    means = df.mean(axis=1)
    mins = df.min(axis=1)
    maxs = df.max(axis=1)

    return xs, means, mins, maxs


# def plot_bi_policy_model(run_name, run_data, ax=None):
#     fig, ax = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
#     fl = ax[0, 0]
#     fa = ax[1, 0]
#     bl = ax[0, 1]
#     ba = ax[1, 1]

#     ls_mapper = LineStyleMapper()
#     c, ls = ls_mapper.get_ls(run_name)

#     xs, means, mins, maxs = window_avg_over_index(run_data, run_data["floss"])
#     fl.plot(xs, means, label=run_name, color=c, linestyle=ls)
#     fl.fill_between(
#         xs,
#         mins,
#         maxs,
#         alpha=0.1,
#         color=c,
#         label=run_name,
#     )

#     xs, means, mins, maxs = window_avg_over_index(run_data, "facc")
#     fa.plot(xs, means, label=run_name, color=c, linestyle=ls)
#     fa.fill_between(
#         xs,
#         mins,
#         maxs,
#         alpha=0.1,
#         color=c,
#         label=run_name,
#     )
#     xs, means, mins, maxs = window_avg_over_index(run_data, "bloss")
#     bl.plot(xs, means, label=run_name, color=c, linestyle=ls)
#     bl.fill_between(
#         xs,
#         mins,
#         maxs,
#         alpha=0.1,
#         color=c,
#         label=run_name,
#     )

#     xs, means, mins, maxs = window_avg_over_index(run_data, "bacc")
#     ba.plot(xs, means, label=run_name, color=c, linestyle=ls)
#     ba.fill_between(
#         xs,
#         mins,
#         maxs,
#         alpha=0.1,
#         color=c,
#         label=run_name,
#     )
#     return fig


# def plot_uni_policy_model(run_name, run_data, ax=None):
#     fig, (fl, fa) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

#     ls_mapper = LineStyleMapper()
#     c, ls = ls_mapper.get_ls(run_name)

#     xs, means, mins, maxs = window_avg_over_index(run_data, "floss")
#     fl.plot(xs, means, label=run_name, color=c, linestyle=ls)
#     fl.fill_between(
#         xs,
#         mins,
#         maxs,
#         alpha=0.1,
#         color=c,
#         label=run_name,
#     )

#     xs, means, mins, maxs = window_avg_over_index(run_data, "facc")
#     fa.plot(xs, means, label=run_name, color=c, linestyle=ls)
#     fa.fill_between(
#         xs,
#         mins,
#         maxs,
#         alpha=0.1,
#         color=c,
#         label=run_name,
#     )
#     return fig


# def plot_bi_heuristic_model(run_name, run_data, ax=None):
#     fig, (fl, bl) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

#     ls_mapper = LineStyleMapper()
#     c, ls = ls_mapper.get_ls(run_name)

#     xs, means, mins, maxs = window_avg_over_index(run_data, "floss")
#     fl.plot(xs, means, label=run_name, color=c, linestyle=ls)
#     fl.fill_between(
#         xs,
#         mins,
#         maxs,
#         alpha=0.1,
#         color=c,
#         label=run_name,
#     )

#     xs, means, mins, maxs = window_avg_over_index(run_data, "bloss")
#     bl.plot(xs, means, label=run_name, color=c, linestyle=ls)
#     bl.fill_between(
#         xs,
#         mins,
#         maxs,
#         alpha=0.1,
#         color=c,
#         label=run_name,
#     )
#     return fig


# def plot_uni_heuristic_model(run_name, run_data, ax=None):
#     fig, fl = plt.subplots(figsize=(12, 10), sharex=True)

#     ls_mapper = LineStyleMapper()
#     c, ls = ls_mapper.get_ls(run_name)

#     xs, means, mins, maxs = window_avg_over_index(run_data, "floss")
#     fl.plot(xs, means, label=run_name, color=c, linestyle=ls)
#     fl.fill_between(
#         xs,
#         mins,
#         maxs,
#         alpha=0.1,
#         color=c,
#         label=run_name,
#     )
#     return fig


def main():
    colors = mpl.colormaps["tab10"].colors
    f = open("socs_stp4cphs.pkl", "rb")
    allruns = pkl.load(f)
    saveroot = Path("figs/socsstp4c/")
    saveroot.mkdir(exist_ok=True, parents=True)
    figkey = ["agent"]
    legendkey = ["share_feature_net", "forward_policy_lr"]
    data = sorted(allruns.values(), key=lambda x: x.args_key(figkey))
    for loss_lr, group in itertools.groupby(data, key=lambda x: x.args_key(figkey)):
        group = list(group)
        group = sorted(group, key=lambda x: x.args_key(legendkey))
        grouped_data = {}
        for runseeds in group:
            legend_name = runseeds.args_key(legendkey)
            grouped_data[legend_name] = runseeds
        fig, ax = plt.subplots(3, 2, sharex=True, figsize=(12, 10))
        fig.suptitle(f"{loss_lr}", size=16)
        ax[0, 0].set_title(f"Train")
        ax[0, 1].set_title(f"Valid")
        ax[0, 0].set_ylabel("Solved", size=14)
        ax[1, 0].set_ylabel("Expanded", size=14)
        ax[2, 0].set_ylabel("Solution length", size=14)

        labels = []
        for i, (legend_name, runseeds) in enumerate(grouped_data.items()):
            rsdata = runseeds.data
            color = colors[i % len(colors)]
            labels.append(legend_name)
            plot_search_vs_time(
                rsdata, "solved", ax=ax[0, 0], color=color, label=legend_name
            )
            plot_search_vs_time(
                rsdata, "exp", ax=ax[1, 0], color=color, label=legend_name
            )
            plot_search_vs_time(
                rsdata, "len", ax=ax[2, 0], color=color, label=legend_name
            )
            ax[2, 0].set_xlabel("Time (s)", size=14)

            plot_valid_vs_time(
                rsdata, "solved", ax=ax[0, 1], color=color, label=legend_name
            )
            plot_valid_vs_time(
                rsdata, "exp", ax=ax[1, 1], color=color, label=legend_name
            )
            plot_valid_vs_time(
                rsdata, "len", ax=ax[2, 1], color=color, label=legend_name
            )
            ax[2, 1].set_xlabel("Time (s)", size=14)
            print(f"Plotted {loss_lr} {legend_name}")

        handles, labels = ax[2, 1].get_legend_handles_labels()
        it = iter(handles)
        handles = [(a, next(it)) for a in it]
        fig.legend(handles, labels[::2])
        fig.tight_layout()
        fig.savefig(saveroot / f"{loss_lr}.pdf", bbox_inches="tight")
        # plt.show()()


if __name__ == "__main__":
    main()
