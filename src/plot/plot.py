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

from utils import LineStyleMapper, PdfTemplate, all_group_key, get_runs_data

cmap = mpl.colormaps["tab20"]
mpl.rcParams["axes.prop_cycle"] = cycler(color=cmap.colors)
mpl.rcParams["axes.linewidth"] = 1
mpl.rc("lines", linewidth=2, linestyle="-")


def plot_all_vs_time(
    runs_data,
    y_data_label: str,
    y_title: str,
    ax=None,
    legend=False,
    batch_size=40,
    window_size=150,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = None
    ls_mapper = LineStyleMapper()
    for run_name, run_data in runs_data.items():
        c, ls = ls_mapper.get_ls(run_name)
        dfs = run_data["search"]
        notna_dfs = []
        for df in dfs:
            df = df[["time", y_data_label]].copy()
            dfg = df.groupby(df.index // batch_size)
            dfs = dfg.aggregate({"time": "max", y_data_label: "mean"})
            dfs["time"] = dfs["time"].cumsum()
            x = dfs.groupby(dfs["time"] // window_size)[y_data_label].mean()
            x.index = x.index.map(lambda x: (x + 1) * window_size)
            notna_dfs.append(x)

        df = pd.concat(notna_dfs, axis=1)
        means = df.mean(axis=1)
        mins = df.min(axis=1)
        maxs = df.max(axis=1)

        ax.plot(df.index.values, means, label=run_name, color=c, linestyle=ls)
        ax.fill_between(
            np.array(df.index.values, dtype=np.float32),
            mins,
            maxs,
            alpha=0.1,
            color=c,
            label=run_name,
        )

    if legend:
        handler, labeler = ax.get_legend_handles_labels()
        it = iter(handler)
        hd = [(a, next(it)) for a in it]
        ax.legend(
            hd, labeler[::2], loc="upper right", bbox_to_anchor=(1.45, 1), fontsize=12
        )

    if fig:
        ax.set_xlabel("Time (s)", size=14)
    ax.set_ylabel(
        f"{y_title}", rotation=0, labelpad=10, size=14, horizontalalignment="right"
    )
    ax.set_title(f"{y_title} vs. time")
    return fig


def plot_all_vs_batch(
    runs_data,
    y_data_label: str,
    y_title: str,
    ax=None,
    legend=False,
    batch_size=40,
    window_size=150,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = None
    ls_mapper = LineStyleMapper()
    for run_name, run_data in runs_data.items():
        c, ls = ls_mapper.get_ls(run_name)
        dfs = run_data["search"]
        batch_dfs = []
        for df in dfs:
            data = df[y_data_label].copy()
            batched_data = data.groupby(data.index // batch_size).mean()
            batched_data.index = range(1, len(batched_data) + 1)
            # x = dfs.groupby(df.index // window_size - 1).mean()
            # x.index = x.index.map(lambda x: (x + 1) * window_size)
            # notna_dfs.append(x)
            batch_dfs.append(batched_data)

        df = pd.concat(batch_dfs, axis=1)
        means = df.mean(axis=1)
        mins = df.min(axis=1)
        maxs = df.max(axis=1)

        ax.plot(df.index.values, means, label=run_name, color=c, linestyle=ls)
        ax.fill_between(
            np.array(df.index.values, dtype=np.float32),
            mins,
            maxs,
            alpha=0.1,
            color=c,
            label=run_name,
        )

    if legend:
        handler, labeler = ax.get_legend_handles_labels()
        it = iter(handler)
        hd = [(a, next(it)) for a in it]
        ax.legend(
            hd, labeler[::2], loc="upper right", bbox_to_anchor=(1.45, 1), fontsize=12
        )

    if fig:
        ax.set_xlabel("Batch", size=14)
    ax.set_ylabel(
        f"{y_title}", rotation=0, labelpad=10, size=14, horizontalalignment="right"
    )
    ax.set_title(f"{y_title} vs. batch")
    return fig


def aggr_runs(runs_data, y_data_label, window_size=5):
    dfs = runs_data["train"]
    aggr_dfs = []
    for df in dfs:
        df = df[y_data_label].copy()
        # df = df.groupby(df.index // window_size).mean()
        aggr_dfs.append(df)

    df = pd.concat(aggr_dfs, axis=1)
    xs = np.array(df.index.values, dtype=np.float32)
    means = df.mean(axis=1)
    mins = df.min(axis=1)
    maxs = df.max(axis=1)

    return xs, means, mins, maxs


def plot_bi_policy_model(run_name, run_data, ax=None):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    fl = ax[0, 0]
    fa = ax[1, 0]
    bl = ax[0, 1]
    ba = ax[1, 1]

    ls_mapper = LineStyleMapper()
    c, ls = ls_mapper.get_ls(run_name)

    xs, means, mins, maxs = aggr_runs(run_data, "floss")
    fl.plot(xs, means, label=run_name, color=c, linestyle=ls)
    fl.fill_between(
        xs,
        mins,
        maxs,
        alpha=0.1,
        color=c,
        label=run_name,
    )

    xs, means, mins, maxs = aggr_runs(run_data, "facc")
    fa.plot(xs, means, label=run_name, color=c, linestyle=ls)
    fa.fill_between(
        xs,
        mins,
        maxs,
        alpha=0.1,
        color=c,
        label=run_name,
    )
    xs, means, mins, maxs = aggr_runs(run_data, "bloss")
    bl.plot(xs, means, label=run_name, color=c, linestyle=ls)
    bl.fill_between(
        xs,
        mins,
        maxs,
        alpha=0.1,
        color=c,
        label=run_name,
    )

    xs, means, mins, maxs = aggr_runs(run_data, "bacc")
    ba.plot(xs, means, label=run_name, color=c, linestyle=ls)
    ba.fill_between(
        xs,
        mins,
        maxs,
        alpha=0.1,
        color=c,
        label=run_name,
    )
    return fig


def plot_uni_policy_model(run_name, run_data, ax=None):
    fig, (fl, fa) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ls_mapper = LineStyleMapper()
    c, ls = ls_mapper.get_ls(run_name)

    xs, means, mins, maxs = aggr_runs(run_data, "floss")
    fl.plot(xs, means, label=run_name, color=c, linestyle=ls)
    fl.fill_between(
        xs,
        mins,
        maxs,
        alpha=0.1,
        color=c,
        label=run_name,
    )

    xs, means, mins, maxs = aggr_runs(run_data, "facc")
    fa.plot(xs, means, label=run_name, color=c, linestyle=ls)
    fa.fill_between(
        xs,
        mins,
        maxs,
        alpha=0.1,
        color=c,
        label=run_name,
    )
    return fig


def plot_bi_heuristic_model(run_name, run_data, ax=None):
    fig, (fl, bl) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ls_mapper = LineStyleMapper()
    c, ls = ls_mapper.get_ls(run_name)

    xs, means, mins, maxs = aggr_runs(run_data, "floss")
    fl.plot(xs, means, label=run_name, color=c, linestyle=ls)
    fl.fill_between(
        xs,
        mins,
        maxs,
        alpha=0.1,
        color=c,
        label=run_name,
    )

    xs, means, mins, maxs = aggr_runs(run_data, "bloss")
    bl.plot(xs, means, label=run_name, color=c, linestyle=ls)
    bl.fill_between(
        xs,
        mins,
        maxs,
        alpha=0.1,
        color=c,
        label=run_name,
    )
    return fig


def plot_uni_heuristic_model(run_name, run_data, ax=None):
    fig, fl = plt.subplots(figsize=(12, 10), sharex=True)

    ls_mapper = LineStyleMapper()
    c, ls = ls_mapper.get_ls(run_name)

    xs, means, mins, maxs = aggr_runs(run_data, "floss")
    fl.plot(xs, means, label=run_name, color=c, linestyle=ls)
    fl.fill_between(
        xs,
        mins,
        maxs,
        alpha=0.1,
        color=c,
        label=run_name,
    )
    return fig


def main():
    exp = "tri4"
    savedir = Path(f"/home/ken/Projects/bilevin/figs/{exp}")
    savedir.mkdir(exist_ok=True, parents=True)
    all_runs_pth = Path("/home/ken/Projects/bilevin/final_runs_test/").glob(f"{exp}*")
    all_runs = get_runs_data(all_runs_pth, all_group_key)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 10))
    plot_all_vs_time(all_runs, "solved", "Solved", ax=ax1, legend=True)
    plot_all_vs_time(all_runs, "exp", "Expanded", ax=ax2)
    plot_all_vs_time(all_runs, "len", "Solution length", ax=ax3)
    ax3.set_xlabel("Time (s)", size=14)
    fig.tight_layout()
    fig.savefig(savedir / f"search_time.pdf", bbox_inches="tight")
    plt.close()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 10))
    plot_all_vs_batch(all_runs, "solved", "Solved", ax=ax1, legend=True)
    plot_all_vs_batch(all_runs, "exp", "Expanded", ax=ax2)
    plot_all_vs_batch(all_runs, "len", "Solution length", ax=ax3)
    ax3.set_xlabel("Batch", size=14)
    fig.tight_layout()
    fig.savefig(savedir / f"search_batch.pdf", bbox_inches="tight")
    plt.close()
    # plt.show()
    for run_name, run_data in all_runs.items():
        if "Bi" in run_name:
            if "AStar" in run_name:
                fig = plot_bi_heuristic_model(run_name, run_data)
            else:
                fig = plot_bi_policy_model(run_name, run_data)
        else:
            if "AStar" in run_name:
                fig = plot_uni_heuristic_model(run_name, run_data)
            else:
                fig = plot_uni_policy_model(run_name, run_data)
        fig.savefig(
            savedir / f"model_{run_name.replace(' ','_')}.pdf", bbox_inches="tight"
        )
        # plt.show()
        plt.close()


if __name__ == "__main__":
    main()
