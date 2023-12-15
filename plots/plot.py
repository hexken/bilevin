import itertools
from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from natsort import natsorted
import numpy as np
import pickle as pkl
import pandas as pd


def plot_single(y, x_label, y_label, title, aggr_size=1, y_min=None, y_max=None):
    fig, ax = plt.subplot()
    if aggr_size != 1:
        max_aggr_idx = (len(y) // aggr_size) * aggr_size
        y = y[:max_aggr_idx]
        y = np.mean(np.array(y).reshape(-1, aggr_size), axis=1)
    x = np.arange(1, len(y) + 1)
    ax.plot(x, y)
    if y_min and y_max:
        ax.set_ylim(y_min, y_max)


def plot_single_ax(ax, y: pd.Series, aggr_size=1, start=1, color=None):
    """Plot a single line on an existing axis, aggregating over aggr_size and ignoring nans."""
    aggr_y = y.astype(np.float64)
    if aggr_size != 1:
        max_aggr_idx = (len(aggr_y) // aggr_size) * aggr_size
        aggr_y = aggr_y[:max_aggr_idx]
        aggr_y = np.array(aggr_y).reshape(-1, aggr_size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            aggr_y = np.nanmean(aggr_y, axis=1)
    x = np.arange(start, start + len(aggr_y))
    mask = np.isfinite(aggr_y)
    if color is not None:
        ax.plot(x[mask], aggr_y[mask], color=color)
    else:
        ax.plot(x[mask], aggr_y[mask])


def plot_valid_single(
    datadir, key1, key2, x_label, y_label, title, y_min=None, y_max=None
):
    all_pds = natsorted(datadir.glob("search_valid_b*.pkl"))
    if key2 is None:
        means = [pkl.load(pd.open("rb"))[key1].mean() for pd in all_pds]
    else:
        means = []
        for pd in all_pds:
            pd = pkl.load(pd.open("rb"))
            means.append((pd[key1] + pd[key2]).mean())
    plot_single(means, x_label, y_label, title, 1, y_min, y_max)


def levin_group_key(pth):
    s = str(pth)
    r = re.search(".*_((Bi)?Levin).*(lr0.\d+)", s)
    if r:
        return " ".join(r.group(1, 3))


def astar_group_key(pth):
    s = str(pth)
    r = re.search(".*_((Bi)?AStar).*(lr0.\d+).*(w\d\.?\d?)", s)
    if r:
        return " ".join(r.group(1, 3, 4))


def plot_runs_separate_seed(run_name, run_paths, batch_size=40):
    # model plots
    if "Bi" in run_name:
        model_fig, model_ax = plt.subplots(
            2,
            2,
        )
        model_ax[0, 1].set_title("Backward")
        model_ax[0, 0].set_title("Forward")
        model_ax[0, 0].set_ylabel("Loss", rotation=0, labelpad=30, size=12)
        model_ax[1, 0].set_ylabel("Accuracy", rotation=0, labelpad=50, size=12)
    else:
        model_fig, model_ax = plt.subplots(
            2,
            1,
        )
        model_ax[0].set_title("Forward")
        model_ax[0].set_ylabel("Loss", rotation=0, labelpad=30, size=12)
        model_ax[1].set_ylabel("Accuracy", rotation=0, labelpad=50, size=12)
    model_fig.suptitle("Model " + run_name)
    model_fig.supxlabel("Batch")
    model_fig.tight_layout()

    # search plots
    if "Bi" in run_name:
        search_fig, search_ax = plt.subplots(6, 1)
        search_ax[4].set_ylabel("F/B Exp", rotation=0, labelpad=30, size=12)
        search_ax[5].set_ylabel("F/B Len", rotation=0, labelpad=30, size=12)
    else:
        search_fig, search_ax = plt.subplots(4, 1)
    search_ax[0].set_ylabel("Solve", rotation=0, labelpad=30, size=12)
    search_ax[1].set_ylabel("Len", rotation=0, labelpad=30, size=12)
    if "Levin" in run_name:
        search_ax[2].set_ylabel(
            "Solution\nprobability.", rotation=0, labelpad=40, size=12
        )
    else:
        search_ax[2].set_ylabel(
            "Avg. start node\nheuristic error (l1)", rotation=0, labelpad=30, size=12
        )
    search_ax[3].set_ylabel("Exp", rotation=0, labelpad=30, size=12)
    search_fig.suptitle("Search " + run_name)
    search_fig.supxlabel("Batch")
    search_fig.tight_layout()

    # valid plots
    if "Bi" in run_name:
        valid_fig, valid_ax = plt.subplots(5, 1)
        valid_ax[3].set_ylabel("F/B Exp", rotation=0, labelpad=30, size=12)
        valid_ax[4].set_ylabel("F/B Len", rotation=0, labelpad=30, size=12)
    else:
        valid_fig, valid_ax = plt.subplots(3, 1)

    valid_ax[0].set_ylabel("Solve", rotation=0, labelpad=30, size=12)
    valid_ax[1].set_ylabel("Len", rotation=0, labelpad=30, size=12)
    valid_ax[2].set_ylabel("Exp", rotation=0, labelpad=30, size=12)
    valid_fig.suptitle("Valid " + run_name)
    valid_fig.supxlabel("Batch")
    valid_fig.tight_layout()

    colors = ["b", "c"]
    for i, pth in enumerate(run_paths):
        color = colors[i]
        search_trains = [
            pkl.load(p.open("rb"))
            for p in natsorted(list(pth.glob("search_train*.pkl")))
        ]
        model_trains = [
            pkl.load(p.open("rb"))
            for p in natsorted(list(pth.glob("model_train*.pkl")))
        ]
        valids = [
            pkl.load(p.open("rb"))
            for p in natsorted(list(pth.glob("search_valid*.pkl")))
        ]

        # model plots
        for model_train in model_trains:
            if "Bi" in run_name:
                plot_single_ax(model_ax[0, 0], model_train["floss"], color=color)
                plot_single_ax(model_ax[1, 0], model_train["facc"], color=color)
                plot_single_ax(model_ax[0, 1], model_train["bloss"], color=color)
                plot_single_ax(model_ax[1, 1], model_train["bacc"], color=color)
            else:
                plot_single_ax(model_ax[0], model_train["floss"], color=color)
                plot_single_ax(model_ax[1], model_train["facc"], color=color)

        # search plots
        for i, search_train in enumerate(search_trains):
            solved = search_train["len"].fillna(0)
            solved[solved > 0] = 1
            assert len(solved) % batch_size == 0
            if i == 0:
                start = 1
            else:
                start = i * (len(search_trains[i - 1]) // batch_size) + 1
            plot_single_ax(
                search_ax[0], solved, aggr_size=batch_size, start=start, color=color
            )
            plot_single_ax(
                search_ax[1],
                search_train["len"],
                aggr_size=batch_size,
                start=start,
                color=color,
            )
            preds = search_train["fpp"]


            if "Bi" in run_name:
                exps = search_train["fexp"] + search_train["bexp"]
            else:
                exps = search_train["fexp"]
            plot_single_ax(
                search_ax[3],
                exps,
                aggr_size=batch_size,
                start=start,
                color=color,
            )
            plot_single_ax(
                search_ax[3],
                search_train["fexp"],
                aggr_size=batch_size,
                start=start,
                color=color,
            )
            # if "Bi" in run_name:
            #     plot_single_ax(search_ax[4], search_train["fbexp"], aggr_size=batch_size)
    plt.show()


def main():
    levin_runs = list(Path("../stp4c_levin").glob("*"))
    astar_runs = list(Path("../stp4c_astar").glob("*"))
    levin_keys = []
    levin_groups = []
    levin_runs = sorted(levin_runs, key=levin_group_key)
    for k, g in itertools.groupby(levin_runs, levin_group_key):
        levin_keys.append(k)
        levin_groups.append(list(g))

    astar_keys = []
    astar_groups = []
    astar_runs = sorted(astar_runs, key=astar_group_key)
    for k, g in itertools.groupby(astar_runs, astar_group_key):
        astar_keys.append(k)
        astar_groups.append(list(g))

    for run_name, run_paths in zip(levin_keys[:1], levin_groups[:1]):
        plot_runs_separate_seed(run_name, run_paths)


if __name__ == "__main__":
    main()
