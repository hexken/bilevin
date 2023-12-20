import itertools
from pathlib import Path
import pickle as pkl
import re
import subprocess
from typing import Optional
import warnings

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import pandas as pd


def plot_single_ax(
    ax, y: pd.Series, aggr_size=1, start=1, xs: Optional[pd.Series] = None, color=None
):
    """Plot a single line on an existing axis, aggregating over aggr_size and ignoring nans."""
    aggr_y = y.astype(np.float64)
    if aggr_size != 1:
        max_aggr_idx = (len(aggr_y) // aggr_size) * aggr_size
        aggr_y = aggr_y[:max_aggr_idx]
        aggr_y = np.array(aggr_y).reshape(-1, aggr_size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            aggr_y = np.nanmean(aggr_y, axis=1)
    if xs is None:
        x = np.arange(start, start + len(aggr_y))
    else:
        x = xs
    mask = np.isfinite(aggr_y)
    if color is not None:
        ax.plot(x[mask], aggr_y[mask], color=color)
    else:
        ax.plot(x[mask], aggr_y[mask])


def levin_group_key(pth):
    s = str(pth)
    r = re.search(".*_((Bi)?Levin).*(lr0.\d+)", s)
    # r = re.search(".*_((Bi)?Levin).*", s)
    if r:
        return " ".join(r.group(1, 3))


def astar_group_key(pth):
    s = str(pth)
    r = re.search(".*_((Bi)?AStar).*(lr0.\d+).*(w\d\.?\d?)", s)
    # r = re.search(".*_((Bi)?AStar).*", s)
    if r:
        return " ".join(r.group(1, 3, 4))


def plot_same_axes(run_name, run_paths, batch_size=40):
    # model plots
    model_fig, model_ax = plt.subplots(2, 2)
    model_ax[0, 1].set_title("Backward")
    model_ax[0, 0].set_title("Forward")
    model_ax[0, 0].set_ylabel("Loss", rotation=0, labelpad=30, size=12)
    model_ax[1, 0].set_ylabel("Accuracy", rotation=0, labelpad=50, size=12)
    model_fig.suptitle("Model " + run_name)
    model_fig.supxlabel("Batch")
    model_fig.tight_layout()

    # search plots
    search_fig, search_ax = plt.subplots(4, 2)
    search_fig.delaxes(search_ax[3, 1])
    search_ax[0, 0].set_ylabel("Solve", rotation=0, labelpad=30, size=12)
    search_ax[1, 0].set_ylabel("Len", rotation=0, labelpad=30, size=12)
    if "Levin" in run_name:
        search_ax[2, 0].set_ylabel(
            "Solution\nprobability", rotation=0, labelpad=40, size=12
        )
    else:
        search_ax[2, 0].set_ylabel(
            "Start node\nheuristic abs. error", rotation=0, labelpad=50, size=12
        )
    search_ax[3, 0].set_ylabel("Exp", rotation=0, labelpad=30, size=12)
    search_ax[0, 1].set_ylabel("F/B Exp", rotation=0, labelpad=30, size=12)
    search_ax[1, 1].set_ylabel("F/B Len", rotation=0, labelpad=30, size=12)
    if "Levin" in run_name:
        search_ax[2, 1].set_ylabel("F/B sol.\nprob.", rotation=0, labelpad=30, size=12)
    else:
        search_ax[2, 1].set_ylabel(
            "F/B start node\nheuristic abs. error", rotation=0, labelpad=60, size=12
        )

    search_fig.suptitle("Search " + run_name)
    search_fig.supxlabel("Batch")
    search_fig.tight_layout()

    # valid plots
    valid_fig, valid_ax = plt.subplots(3, 2)
    valid_fig.delaxes(valid_ax[2, 1])
    valid_ax[0, 0].set_ylabel("Solve", rotation=0, labelpad=30, size=12)
    valid_ax[1, 0].set_ylabel("Len", rotation=0, labelpad=30, size=12)
    valid_ax[2, 0].set_ylabel("Exp", rotation=0, labelpad=30, size=12)
    valid_ax[0, 1].set_ylabel("F/B Exp", rotation=0, labelpad=30, size=12)
    valid_ax[1, 1].set_ylabel("F/B Len", rotation=0, labelpad=30, size=12)
    valid_fig.suptitle("Valid " + run_name)
    valid_fig.supxlabel("Valid run")
    valid_fig.tight_layout()

    colors = ["b", "c"]
    for pth in run_paths:
        color = colors.pop()
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

        # fill model plots
        for i, model_train in enumerate(model_trains):
            xs = model_train.index.values
            # print(xs)
            plot_single_ax(model_ax[0, 0], model_train["floss"], xs=xs, color=color)
            if "Levin" in run_name:
                plot_single_ax(model_ax[1, 0], model_train["facc"], xs=xs, color=color)
            if "Bi" in run_name:
                plot_single_ax(model_ax[0, 1], model_train["bloss"], xs=xs, color=color)
                if "Levin" in run_name:
                    plot_single_ax(
                        model_ax[1, 1], model_train["bacc"], xs=xs, color=color
                    )

        # fill search plots
        for i, search_train in enumerate(search_trains):
            # offset batch axis by stage
            if i == 0:
                start = 1
            else:
                start = i * (len(search_trains[i - 1]) // batch_size) + 1

            # plot solved
            solved = search_train["len"].fillna(0)
            solved[solved > 0] = 1
            assert len(solved) % batch_size == 0
            plot_single_ax(
                search_ax[0, 0], solved, aggr_size=batch_size, start=start, color=color
            )
            # plot Len
            plot_single_ax(
                search_ax[1, 0],
                search_train["len"],
                aggr_size=batch_size,
                start=start,
                color=color,
            )

            # plot preds
            if "Levin" in run_name:
                preds = np.exp(-search_train["fpp"])
            else:
                preds = search_train["fpp"]
            plot_single_ax(
                search_ax[2, 0],
                preds,
                aggr_size=batch_size,
                start=start,
                color=color,
            )

            # plot exp
            if "Bi" in run_name:
                exps = search_train["fexp"] + search_train["bexp"]
            else:
                exps = search_train["fexp"]
            plot_single_ax(
                search_ax[3, 0],
                exps,
                aggr_size=batch_size,
                start=start,
                color=color,
            )
            if "Bi" in run_name:
                # plot F/B exp
                fb_exps = search_train["fexp"] / search_train["bexp"]
                plot_single_ax(
                    search_ax[0, 1],
                    fb_exps,
                    aggr_size=batch_size,
                    start=start,
                    color=color,
                )

                # plot F/B len
                fb_lens = search_train["fg"] / search_train["bg"]
                plot_single_ax(
                    search_ax[1, 1],
                    fb_lens,
                    aggr_size=batch_size,
                    start=start,
                    color=color,
                )

                # plot F/B preds
                if "Levin" in run_name:
                    fpreds = np.exp(-search_train["fpp"])
                    bpreds = np.exp(-search_train["bpp"])
                else:
                    fpreds = search_train["fpp"]
                    bpreds = search_train["bpp"]
                fb_pp = fpreds / bpreds
                plot_single_ax(
                    search_ax[2, 1],
                    fb_pp,
                    aggr_size=batch_size,
                    start=start,
                    color=color,
                )

        # fill valid plots
        # plot solved
        valid_solved = []
        valid_lens = []
        valid_exps = []
        valid_fb_lens = []
        valid_fb_exps = []
        for search_valid in valids:
            solved = search_valid["len"].fillna(0)
            solved[solved > 0] = 1
            assert len(solved) % batch_size == 0
            valid_solved.append(solved.mean())

            valid_lens.append(search_valid["len"].mean())

            if "Bi" in run_name:
                exps = search_valid["fexp"] + search_valid["bexp"]
                fb_exps = search_valid["fexp"] / search_valid["bexp"]
                fb_lens = search_valid["fg"] / search_valid["bg"]
                valid_fb_exps.append(fb_exps.mean())
                valid_fb_lens.append(fb_lens.mean())
            else:
                exps = search_valid["fexp"]
            valid_exps.append(exps.mean())

        # plot solved
        plot_single_ax(valid_ax[0, 0], pd.Series(valid_solved), color=color)

        # plot Len
        plot_single_ax(valid_ax[1, 0], pd.Series(valid_lens), color=color)

        # plot exp
        plot_single_ax(valid_ax[2, 0], pd.Series(valid_exps), color=color)

        # plot fb exp and len
        if "Bi" in run_name:
            # plot F/B exp
            plot_single_ax(valid_ax[0, 1], pd.Series(valid_fb_exps), color=color)

            # plot F/B len
            plot_single_ax(valid_ax[1, 1], pd.Series(valid_fb_lens), color=color)

    return model_fig, search_fig, valid_fig


def group_and_plot(path, group_key_func, batch_size=40):
    runpaths = list(path.glob("*"))
    print(runpaths)
    runnames = []
    rundata = []
    runs = sorted(runpaths, key=group_key_func)
    for k, g in itertools.groupby(runs, group_key_func):
        runnames.append(k)
        rundata.append(list(g))

    figs = []
    for run_name, run_paths in zip(runnames, rundata):
        model_fig, search_fig, valid_fig = plot_same_axes(
            run_name, run_paths, batch_size
        )
        figs.append((run_name, model_fig, search_fig, valid_fig))

    return figs


class PdfTemplate:
    def __init__(self, figs, filename, toc=True):
        self.figs = figs
        self.toc = toc
        self.filename = filename
        self.text = []

    def render(self):
        self._pagebreak()
        for fig in self.figs:
            self._h1(fig.split(".")[0])
            self._img(os.path.join("fig", fig))
            self._pagebreak()
        self.text = "\n\n".join(self.text)

    def export(self):
        md_file = f"{self.filename}.md"
        pdf_file = f"{self.filename}.pdf"
        pandoc = ["pandoc", f"{md_file}", f"-o {pdf_file}"]
        with open(md_file, "w") as f:
            f.write(self.text)
        if self.toc:
            pandoc.append("--toc")
        os.system(" ".join(pandoc))

    def _pagebreak(self):
        self.text.append("\pagebreak")

    def _h1(self, text):
        self.text.append(f"# {text}")

    def _img(self, img):
        self.text.append(f"![]({img})")


def main():
    levin_runs = Path("../stp4c_sweep/stp4c_levin")
    astar_runs = Path("../stp4c_sweep/stp4c_astar")
    # astar_noclip_runs = Path("../sweep_runs/stp4c_astar_noclip")

    levin_figs = group_and_plot(levin_runs, levin_group_key, batch_size=40)
    astar_figs = group_and_plot(astar_runs, astar_group_key, batch_size=40)
    figs = levin_figs + astar_figs

    fig_dir = Path("fig").mkdir(exist_ok=True)
    figs = natsorted(figs, key=lambda x: x[0])
    for run_name, (model_fig, search_fig, valid_fig) in figs:
        model_fig.savefig(f"fig/{run_name.replace(' ', '_')}_model.png")
        search_fig.savefig(f"fig/{run_name.replace(' ', '_')}_search.png")
        valid_fig.savefig(f"fig/{run_name.replace(' ', '_')}_valid.png")


if __name__ == "__main__":
    main()
