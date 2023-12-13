import itertools
from pathlib import Path
import re

import matplotlib.pyplot as plt
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


def plot_single_ax(ax, y, aggr_size=40):
    """Plot a single line on an existing axis, aggregating over aggr_size and ignoring nans."""
    y = np.array(pd.to_numeric(y, errors="coerce"))
    if aggr_size != 1:
        max_aggr_idx = (len(y) // aggr_size) * aggr_size
        y = y[:max_aggr_idx]
        y = np.array(y).reshape(-1, aggr_size)
        y = np.nanmean(y, axis=1)
    else:
        y = np.nanmean(y, axis=1)
    x = np.arange(1, len(y) + 1)
    print(y)
    mask = np.isfinite(y)
    ax.plot(x[mask], y[mask])


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
        model_fig, model_ax = plt.subplots(2, 2, sharex=True)
        model_ax[0, 1].set_title("Backward")
        model_ax[0, 0].set_title("Forward")
        model_ax[0, 0].set_ylabel("Loss", rotation=0, labelpad=30, size=12)
        model_ax[1, 0].set_ylabel("Accuracy", rotation=0, labelpad=50, size=12)
    else:
        model_fig, model_ax = plt.subplots(2, 1, sharex=True)
        model_ax[0].set_title("Forward")
        model_ax[0].set_ylabel("Loss", rotation=0, labelpad=30, size=12)
        model_ax[1].set_ylabel("Accuracy", rotation=0, labelpad=50, size=12)
    model_fig.suptitle("Model " + run_name)
    model_fig.supxlabel("Batch")
    model_fig.tight_layout()

    # search plots
    if "Bi" in run_name:
        search_fig, search_ax = plt.subplots(5, 1, sharex=True)
        search_ax[4].set_ylabel("F/B Exp", rotation=0, labelpad=30, size=12)
    else:
        search_fig, search_ax = plt.subplots(4, 1, sharex=True)
    search_ax[0].set_ylabel("Solve", rotation=0, labelpad=30, size=12)
    search_ax[1].set_ylabel("Len", rotation=0, labelpad=30, size=12)
    if "Levin" in run_name:
        search_ax[2].set_ylabel("Sol. prob.", rotation=0, labelpad=30, size=12)
    else:
        search_ax[2].set_ylabel("Start heur.", rotation=0, labelpad=30, size=12)
    search_ax[3].set_ylabel("Exp", rotation=0, labelpad=30, size=12)
    search_fig.suptitle("Search " + run_name)
    search_fig.supxlabel("Batch")
    search_fig.tight_layout()

    # valid plots
    if "Bi" in run_name:
        valid_fig, valid_ax = plt.subplots(4, 1, sharex=True)
        valid_ax[3].set_ylabel("F/B Exp", rotation=0, labelpad=30, size=12)
    else:
        valid_fig, valid_ax = plt.subplots(3, 1, sharex=True)
    valid_ax[0].set_ylabel("Solve", rotation=0, labelpad=30, size=12)
    valid_ax[1].set_ylabel("Len", rotation=0, labelpad=30, size=12)
    valid_ax[2].set_ylabel("Exp", rotation=0, labelpad=30, size=12)
    valid_fig.suptitle("Valid " + run_name)
    valid_fig.supxlabel("Batch")
    valid_fig.tight_layout()

    for pth in run_paths:
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
                plot_single_ax(model_ax[0, 0], model_train["floss"])
                plot_single_ax(model_ax[1, 0], model_train["facc"])
                plot_single_ax(model_ax[0, 1], model_train["bloss"])
                plot_single_ax(model_ax[1, 1], model_train["bacc"])
            else:
                plot_single_ax(model_ax[0], model_train["floss"])
                plot_single_ax(model_ax[1], model_train["facc"])
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
