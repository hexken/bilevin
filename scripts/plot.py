import itertools
from pathlib import Path
import pickle as pkl
from typing import Optional

from cycler import cycler
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from natsort import natsorted
import plotting.keys as pkeys
from plotting.utils import RunSeeds
from plotting.utils import LineStyleMapper, PdfTemplate, get_runs_data

cmap = mpl.colormaps["tab20"]
mpl.rcParams["axes.prop_cycle"] = cycler(color=cmap.colors)
mpl.rcParams["axes.linewidth"] = 1
mpl.rc("lines", linewidth=1, linestyle="-")


def plot_search(
    run_data,
    y_data_label: str,
    ax,
    style,
    label=None,
    batch_size=4,
    n_final_stage_epochs=25,
):
    # plot seeds corresponding to a run
    dfs = run_data["search"]
    notna_dfs = []
    for df in dfs:
        dfg = df[["stage", "epoch", y_data_label]]
        dfg = dfg.groupby(df["batch"] // batch_size)
        dfg = dfg.aggregate({"stage": "max", "epoch": "max", y_data_label: "mean"})
        dfg = dfg.groupby(["stage", "epoch"], as_index=True).mean()
        notna_dfs.append(dfg)

    df = pd.concat(notna_dfs, axis=1)
    central = df.median(axis=1)
    lower = df.min(axis=1)
    upper = df.max(axis=1)
    maxs = max(s for s, _ in df.index.values)
    xlabels = [f"s{s}e{e}" if s >= maxs else f"s{s}" for s, e in df.index.values]
    xticks = np.arange(1, len(xlabels) + 1)

    color, linestyle, hatch = style
    ax.plot(xticks, central, color=color, linestyle=linestyle, label=label)
    ax.fill_between(
        xticks,
        lower,
        upper,
        edgecolor=(*color, 0.1),
        hatch=hatch,
        facecolor=(*color, 0.1),
        label=label,
    )
    ax.set_xticks(xticks, xlabels, rotation=70)


def plot_valid(
    run_data,
    y_data_label: str,
    ax,
    style,
    label=None,
):
    # plot seeds corresponding to a run
    dfs = run_data["valid"]
    notna_dfs = []
    min_n_vals = dfs[0]["batch"].nunique()
    for df in dfs:
        dfg = df.groupby(df["batch"], as_index=False)
        dfg = dfg.aggregate({y_data_label: "mean"})[y_data_label]
        if len(dfg) != min_n_vals:
            print(f"Warning: incomplete run detected: {len(df)} != {min_n_vals}")
            min_n_vals = len(dfg)
        notna_dfs.append(dfg)

    # notna_dfs = [df.iloc[:min_n_vals] for df in notna_dfs]
    df = pd.concat(notna_dfs, axis=1)
    central = df.median(axis=1)
    lower = df.min(axis=1)
    upper = df.max(axis=1)
    n_valids = np.arange(1, min_n_vals + 1)

    color, linestyle, hatch = style
    ax.plot(n_valids, central, color=color, linestyle=linestyle, label=label)
    ax.fill_between(
        n_valids,
        lower,
        upper,
        edgecolor=(*color, 0.1),
        hatch=hatch,
        facecolor=(*color, 0.1),
        label=label,
    )


def plot_search_vs_batch(
    run_data,
    y_data_label: str,
    ax,
    style,
    label=None,
    batch_size=4,
    n_final_stage_epochs=25,
    window_size=100,
):
    # plot seeds corresponding to a run
    dfs = run_data["search"]
    notna_dfs = []
    for df in dfs:
        dfg = df[y_data_label]
        dfg = dfg.groupby(df["batch"] // batch_size, as_index=True).mean()
        dfg = dfg.rolling(window_size, min_periods=1).mean()
        # dfg = dfg.groupby(["stage", "epoch"], as_index=True).mean()
        notna_dfs.append(dfg)

    df = pd.concat(notna_dfs, axis=1)
    central = df.median(axis=1)
    lower = df.min(axis=1)
    upper = df.max(axis=1)
    xs = df.index.values

    color, linestyle, hatch = style
    ax.plot(xs, central, color=color, linestyle=linestyle, label=label)
    ax.fill_between(
        xs,
        lower,
        upper,
        edgecolor=(*color, 0.1),
        hatch=hatch,
        facecolor=(*color, 0.1),
        label=label,
    )


def plot_search_vs_time(
    run_data,
    y_data_label: str,
    ax,
    color=None,
    linestyle=None,
    label=None,
    batch_size=4,
    window_size=200,
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
    means = df.median(axis=1)
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


def window_avgs(data: list[pd.DataFrame], y_data_label, window_size=100):
    aggr_dfs = []
    max_batch = max(df.index.values[-1] for df in data)
    for df in data:
        df = df[y_data_label]
        df = df.reindex(range(1, max_batch + 1), fill_value=np.nan)
        df = df.rolling(window_size, min_periods=1).mean()
        aggr_dfs.append(df)

    xs = df.index.values
    df = pd.concat(aggr_dfs, axis=1)
    means = df.median(axis=1)
    mins = df.min(axis=1)
    maxs = df.max(axis=1)

    return xs, means, mins, maxs


def plot_medians(ax, run_data, y_label, color):
    xs, central, lower, upper = window_avgs(run_data, y_label)
    ax.plot(xs, central, color=color)
    ax.fill_between(
        xs,
        lower,
        upper,
        edgecolor=(*color, 0.1),
        facecolor=(*color, 0.1),
    )


def plot_bi_policy_model(run_data: dict, label: str):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    fl = ax[0, 0]
    fl.set_title(f"Forward")
    fl.set_ylabel("Loss")
    fa = ax[1, 0]
    fa.set_ylabel("Acc")
    fa.set_xlabel("Batch")

    bl = ax[0, 1]
    bl.set_title(f"Backward")
    ba = ax[1, 1]
    ba.set_xlabel("Batch")

    ls_mapper = LineStyleMapper()
    color, _, _ = ls_mapper.get_ls(label)

    plot_medians(fl, run_data["train"], "floss", color)
    plot_medians(fa, run_data["train"], "facc", color)
    plot_medians(bl, run_data["train"], "bloss", color)
    plot_medians(ba, run_data["train"], "bacc", color)
    return fig, ax


def plot_uni_policy_model(run_data: dict, label: str):
    fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fl = ax[0]
    fl.set_title(f"Forward")
    fl.set_ylabel("Loss")
    fa = ax[1]
    fa.set_xlabel("Batch")

    ls_mapper = LineStyleMapper()
    color, _, _ = ls_mapper.get_ls(label)

    plot_medians(fl, run_data["train"], "floss", color)
    plot_medians(fa, run_data["train"], "facc", color)
    return fig, ax


def plot_bi_heuristic_model(run_data: dict, label: str):
    fig, ax = plt.subplots(1, 2, figsize=(12, 10), sharex=True)
    fl = ax[0]
    fl.set_title(f"Forward")
    fl.set_ylabel("Loss")
    fl.set_xlabel("Batch")
    bl = ax[1]
    bl.set_title(f"Backward")
    bl.set_xlabel("Batch")

    ls_mapper = LineStyleMapper()
    color, _, _ = ls_mapper.get_ls(label)

    plot_medians(fl, run_data["train"], "floss", color)
    plot_medians(bl, run_data["train"], "bloss", color)
    return fig, ax


def plot_uni_heuristic_model(run_data: dict, label: str):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), sharex=True)
    fl = ax
    fl.set_title(f"Forward")
    fl.set_ylabel("Loss")
    fl.set_xlabel("Batch")
    ls_mapper = LineStyleMapper()
    color, _, _ = ls_mapper.get_ls(label)

    plot_medians(fl, run_data["train"], "floss", color)
    return fig, ax


def plot_bi_policy_heuristic_model(run_data: dict, label: str):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    fl = ax[0, 0]
    fl.set_title(f"Forward")
    fl.set_ylabel("Loss")
    fa = ax[1, 0]
    fa.set_ylabel("Acc")
    fa.set_xlabel("Batch")
    # flh = ax[2, 1]

    bl = ax[0, 1]
    bl.set_title(f"Backward")
    ba = ax[1, 1]
    ba.set_xlabel("Batch")
    # blh = ax[2, 1]

    ls_mapper = LineStyleMapper()
    color, _, _ = ls_mapper.get_ls(label)

    # flp.set_ylabel("Policy loss", size=14)
    plot_medians(fl, run_data["train"], "floss", color)
    plot_medians(fa, run_data["train"], "facc", color)
    # plot_medians(flh, run_data["train"], "bloss", color)

    plot_medians(bl, run_data["train"], "bloss", color)
    plot_medians(ba, run_data["train"], "bacc", color)
    # plot_medians(blh, run_data["train"], "bacc", color)
    return fig, ax


def plot_uni_policy_heuristic_model(run_data: dict, label: str):
    fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    flp = ax[0]
    flp.set_title(f"Forward")
    flp.set_ylabel("Loss")
    fa = ax[1]
    fa.set_ylabel("Acc")
    fa.set_xlabel("Batch")
    # flh = ax[2]

    ls_mapper = LineStyleMapper()
    color, _, _ = ls_mapper.get_ls(label)

    plot_medians(flp, run_data["train"], "floss", color)
    plot_medians(fa, run_data["train"], "facc", color)
    # plot_medians(flh, run_data["train"], "bloss", color)

    return fig, ax


def plot_domain(run_data: dict, outdir: str):
    saveroot = Path(outdir)
    saveroot.mkdir(exist_ok=True, parents=True)
    figkey = ["problems_path"]
    legendkey = ["agent"]
    data = sorted(run_data.values(), key=lambda x: x.args_key(figkey))
    ls_mapper = LineStyleMapper()
    # todo they should all have same prob path anyway
    for fkey, group in itertools.groupby(data, key=lambda x: x.args_key(figkey)):
        fkey = Path(fkey).parent.name
        group = list(group)
        group = sorted(group, key=lambda x: x.args_key(legendkey))
        grouped_data = {}
        for runseeds in group:
            agent = runseeds.args_key(legendkey)
            grouped_data[agent] = runseeds

        # search and val
        fig, ax = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle(f"{fkey}", size=16)
        ax[0, 0].set_title(f"Train")
        ax[0, 1].set_title(f"Valid")
        ax[0, 0].set_ylabel("Solved", size=14)
        ax[1, 0].set_ylabel("Expanded", size=14)
        ax[2, 0].set_ylabel("Solution length", size=14)

        # search vs batch
        fig2, ax2 = plt.subplots(3, 1, figsize=(12, 10))
        fig2.suptitle(f"{fkey}", size=16)
        ax2[0].set_title(f"Train")
        ax2[0].set_ylabel("Solved", size=14)
        ax2[1].set_ylabel("Expanded", size=14)
        ax2[2].set_ylabel("Solution length", size=14)

        labels = []
        # loop over agents (uni/bi - levin/phs/astar, etc.)
        for agent, runseeds in grouped_data.items():
            style = ls_mapper.get_ls(agent)
            # fill search and val plots
            rsdata = runseeds.data
            labels.append(agent)
            plot_search(
                rsdata,
                "solved",
                ax=ax[0, 0],
                style=style,
                label=agent,
            )
            plot_search(rsdata, "exp", ax=ax[1, 0], style=style, label=agent)
            plot_search(rsdata, "len", ax=ax[2, 0], style=style, label=agent)

            plot_valid(
                rsdata,
                "solved",
                ax=ax[0, 1],
                style=style,
                label=agent,
            )
            plot_valid(rsdata, "exp", ax=ax[1, 1], style=style, label=agent)
            plot_valid(rsdata, "len", ax=ax[2, 1], style=style, label=agent)
            ax[2, 1].set_xlabel("Valid #", size=14)

            # fill search vs batch plots
            plot_search_vs_batch(
                rsdata,
                "solved",
                ax=ax2[0],
                style=style,
                label=agent,
            )
            plot_search_vs_batch(rsdata, "exp", ax=ax2[1], style=style, label=agent)
            plot_search_vs_batch(rsdata, "len", ax=ax2[2], style=style, label=agent)
            # make model train plots
            if "Bi" in agent:
                if "PHS" in agent:
                    f, a = plot_bi_policy_heuristic_model(rsdata, agent)
                elif "Levin" in agent:
                    f, a = plot_bi_policy_model(rsdata, agent)
                elif "AStar" in agent:
                    f, a = plot_bi_heuristic_model(rsdata, agent)
                else:
                    raise ValueError(f"Unknown agent: {agent}")
            else:
                if "PHS" in agent:
                    f, a = plot_uni_policy_heuristic_model(rsdata, agent)
                elif "Levin" in agent:
                    f, a = plot_uni_policy_model(rsdata, agent)
                elif "AStar" in agent:
                    f, a = plot_uni_heuristic_model(rsdata, agent)
                else:
                    raise ValueError(f"Unknown agent: {agent}")
            f.savefig(saveroot / f"{agent}_model.pdf", bbox_inches="tight")
            plt.close()
            print(f"Plotted {fkey} {agent}")

        handles, labels = ax[2, 1].get_legend_handles_labels()
        it = iter(handles)
        handles = [(a, next(it)) for a in it]
        fig.legend(handles, labels[::2])
        fig.tight_layout()
        fig.savefig(saveroot / f"search_valid.pdf", bbox_inches="tight")
        plt.close()

        fig2.tight_layout()
        fig2.savefig(saveroot / f"search_vs_batch.pdf", bbox_inches="tight")
        plt.close()


def main():
    colors = mpl.colormaps["tab10"].colors
    dom_paths = list(Path("data/thes/runs_data/").glob("*.pkl"))
    print("Found domains:")
    for dom in dom_paths:
        print(dom.stem)

    for dom in dom_paths:
        dom_data = pkl.load(dom.open("rb"))
        plot_domain(dom_data, f"figs/thes/{dom.stem}")


if __name__ == "__main__":
    main()
