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

from plotting.utils import (
    LineStyleMapper,
    PdfTemplate,
    all_group_key,
    get_runs_data,
    phs_test_key,
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

        dfg = df.groupby(df["batch"])[y_data_label].mean()
        notna_dfs.append(dfg)

    df = pd.concat(notna_dfs, axis=1)
    xs = np.array(xs).max(axis=0)
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


def opt_loss_group_key(item):
    key, val = item
    words = key.split()
    return f"{words[-1]} {words[1]} {words[3]}"


def lr_mom_group_key(item):
    key, val = item
    words = key.split()
    return f"{words[2]}"


def main():
    colors = mpl.colormaps["tab10"].colors
    f = open("all_runs2.pkl", "rb")
    allruns = pkl.load(f)
    saveroot = Path("figs/figstest2")
    data = sorted(allruns.items(), key=opt_loss_group_key)
    for opt_loss, group in itertools.groupby(data, key=opt_loss_group_key):
        print(opt_loss)
        saveoptloss = saveroot / opt_loss.replace(" ", "_")
        saveoptloss.mkdir(exist_ok=True, parents=True)
        # data = {}
        group = list(group)
        group = sorted(group, key=lr_mom_group_key)
        for lr, lrgroup in itertools.groupby(group, key=lr_mom_group_key):
            print(lr)
            savelr = saveoptloss / lr.replace(" ", "_")
            savelr.mkdir(exist_ok=True, parents=True)
            data = {}
            for lg in lrgroup:
                words = lg[0].split()
                legend_name = f"{words[4]} {words[5]}"
                data[legend_name] = lg[1]
            fig, ax = plt.subplots(3, 2, sharex=True, figsize=(12, 10))
            fig.suptitle(f"{opt_loss} {lr}", size=16)
            ax[0, 0].set_title(f"Train")
            ax[0, 1].set_title(f"Valid")
            ax[0, 0].set_ylabel("Solved", size=14)
            ax[1, 0].set_ylabel("Expanded", size=14)
            ax[2, 0].set_ylabel("Solution length", size=14)

            labels = []
            for i, (pname, pdata) in enumerate(data.items()):
                color = colors[i % len(colors)]
                labels.append(pname)
                plot_search_vs_time(
                    pdata, "solved", ax=ax[0, 0], color=color, label=pname
                )
                plot_search_vs_time(pdata, "exp", ax=ax[1, 0], color=color, label=pname)
                plot_search_vs_time(pdata, "len", ax=ax[2, 0], color=color, label=pname)
                ax[2, 0].set_xlabel("Time (s)", size=14)

                plot_valid_vs_time(
                    pdata, "solved", ax=ax[0, 1], color=color, label=pname
                )
                plot_valid_vs_time(pdata, "exp", ax=ax[1, 1], color=color, label=pname)
                plot_valid_vs_time(pdata, "len", ax=ax[2, 1], color=color, label=pname)
                ax[2, 1].set_xlabel("Time (s)", size=14)

            handles, labels = ax[2, 1].get_legend_handles_labels()
            it = iter(handles)
            handles = [(a, next(it)) for a in it]
            fig.legend(handles, labels[::2])
            # fig.tight_layout()
            fig.savefig(savelr / f"time.pdf", bbox_inches="tight")
            # plt.show()()

    # exp = "stp3"
    # savedir = Path(f"/home/ken/Projects/bilevin/stp3figs/{exp}")
    # savedir.mkdir(exist_ok=True, parents=True)
    # all_runs_pth = Path("/home/ken/Projects/bilevin/stp3_phs/")
    # all_runs = get_runs_data(all_runs_pth, phs_test_key)

    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 10))
    # plot_all_vs_time(all_runs, "solved", "Solved", ax=ax1, legend=True)
    # plot_all_vs_time(all_runs, "exp", "Expanded", ax=ax2)
    # plot_all_vs_time(all_runs, "len", "Solution length", ax=ax3)
    # ax3.set_xlabel("Time (s)", size=14)
    # fig.tight_layout()
    # fig.savefig(savedir / f"search_time.pdf", bbox_inches="tight")
    # plt.close()

    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 10))
    # plot_all_vs_batch(all_runs, "solved", "Solved", ax=ax1, legend=True)
    # plot_all_vs_batch(all_runs, "exp", "Expanded", ax=ax2)
    # plot_all_vs_batch(all_runs, "len", "Solution length", ax=ax3)
    # ax3.set_xlabel("Batch", size=14)
    # fig.tight_layout()
    # fig.savefig(savedir / f"search_batch.pdf", bbox_inches="tight")
    # plt.close()
    # # plt.show()
    # for run_name, run_data in all_runs.items():
    #     if "Bi" in run_name:
    #         if "AStar" in run_name:
    #             fig = plot_bi_heuristic_model(run_name, run_data)
    #         else:
    #             fig = plot_bi_policy_model(run_name, run_data)
    #     else:
    #         if "AStar" in run_name:
    #             fig = plot_uni_heuristic_model(run_name, run_data)
    #         else:
    #             fig = plot_uni_policy_model(run_name, run_data)
    #     fig.savefig(
    #         savedir / f"model_{run_name.replace(' ','_')}.pdf", bbox_inches="tight"
    #     )
    #     # plt.show()
    #     plt.close()


if __name__ == "__main__":
    main()
