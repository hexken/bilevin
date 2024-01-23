import itertools
import matplotlib.colors as mcolors
import json
from pathlib import Path
import pickle as pkl
import re
import subprocess
from typing import Optional
import warnings

import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import pandas as pd


def plot_single_ax(
    ax,
    y: pd.Series,
    aggr_size=1,
    start=1,
    xs: Optional[pd.Series] = None,
    stage_vlines=None,
    epoch_vlines=None,
    color=None,
):
    """Plot a single line on an existing axis, aggregating over aggr_size and ignoring nans."""
    aggr_y = y.astype("Float64").to_numpy(dtype=np.float64, na_value=np.nan)
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


def add_vlines(ax, stage_vlines=None, epoch_vlines=None):
    if stage_vlines is not None:
        for vline in stage_vlines:
            ax.axvline(vline, color="k", linestyle=":", linewidth=0.1)
    if epoch_vlines is not None:
        for vline in epoch_vlines:
            ax.axvline(vline, color="k", linestyle="-", linewidth=0.15)


def levin_group_key(pth):
    s = str(pth)
    r = re.search(".*_((Bi)?Levin).*(lr0.\d+)", s)
    # r = re.search(".*_((Bi)?Levin)", s)  # ".*(lr0.\d+)", s)
    # r = re.search(".*_((Bi)?Levin).*", s)
    if r:
        return " ".join(r.group(1, 3))
        # return " ".join(r.group(1))


def astar_group_key(pth):
    s = str(pth)
    r = re.search(".*_((Bi)?AStar).*(lr0.\d+).*(w\d\.?\d?)", s)
    # r = re.search(".*_((Bi)?AStar).*", s)
    if r:
        return " ".join(r.group(1, 3, 4))


def plot_same_axes(run_name, run_paths: list, batch_size=40):
    # model plots
    model_fig, model_ax = plt.subplots(2, 2, tight_layout=True, figsize=(8, 11))
    model_ax[0, 1].set_title("Backward")
    model_ax[0, 0].set_title("Forward")
    model_ax[0, 0].set_ylabel("Loss", rotation=0, labelpad=30, size=12)
    model_ax[1, 0].set_ylabel("Accuracy", rotation=0, labelpad=50, size=12)
    # model_fig.suptitle("Model " + run_name)
    model_fig.supxlabel("Batch")

    # search plots
    search_fig, search_ax = plt.subplots(4, 2, tight_layout=True, figsize=(8, 11))
    search_fig.delaxes(search_ax[3, 1])
    search_ax[0, 0].set_ylabel("Solve", rotation=0, labelpad=30, size=12)
    search_ax[1, 0].set_ylabel("Len", rotation=0, labelpad=30, size=12)
    if "Levin" in run_name:
        search_ax[2, 0].set_ylabel(
            "Solution\nprobability", rotation=0, labelpad=40, size=12
        )
    else:
        search_ax[2, 0].set_ylabel(
            "Start node\nheuristic abs.\n error", rotation=0, labelpad=50, size=12
        )
    search_ax[3, 0].set_ylabel("Exp", rotation=0, labelpad=30, size=12)
    search_ax[0, 1].set_ylabel("F/B Exp", rotation=0, labelpad=30, size=12)
    search_ax[1, 1].set_ylabel("F/B Len", rotation=0, labelpad=30, size=12)
    if "Levin" in run_name:
        search_ax[2, 1].set_ylabel("F/B sol.\nprob.", rotation=0, labelpad=30, size=12)
    else:
        search_ax[2, 1].set_ylabel(
            "F/B start node\nheuristic abs.\n error", rotation=0, labelpad=60, size=12
        )

    # search_fig.suptitle("Search " + run_name)
    search_fig.supxlabel("Batch")

    # valid plots
    valid_fig, valid_ax = plt.subplots(3, 2, tight_layout=True, figsize=(8, 11))
    valid_fig.delaxes(valid_ax[2, 1])
    valid_ax[0, 0].set_ylabel("Solve", rotation=0, labelpad=30, size=12)
    valid_ax[1, 0].set_ylabel("Len", rotation=0, labelpad=30, size=12)
    valid_ax[2, 0].set_ylabel("Exp", rotation=0, labelpad=30, size=12)
    valid_ax[0, 1].set_ylabel("F/B Exp", rotation=0, labelpad=30, size=12)
    valid_ax[1, 1].set_ylabel("F/B Len", rotation=0, labelpad=30, size=12)
    # valid_fig.suptitle("Valid " + run_name)
    valid_fig.supxlabel("Valid run")

    colors = ["b", "c", "g", "m"]

    # find longest run, use to compute vlines
    longest_run, longest_search = 0, 0
    for i, pth in enumerate(run_paths):
        search_len = len(list(pth.glob("search_train*.pkl")))
        if search_len > longest_search:
            longest_search = search_len
            longest_run = i

    search_stages = [
        p
        for p in run_paths[longest_run].glob("search_train_*.pkl")
        if "_e" not in p.name
    ]
    n_search_stages = len(search_stages)
    print(search_stages[0])
    ssbi = len(pkl.load(search_stages[0].open("rb"))) // batch_size
    stage_vlines = [ssbi * i for i in range(1, n_search_stages + 1)]

    search_epochs = list(run_paths[longest_run].glob("search_train*_e*.pkl"))
    n_search_epochs = len(search_epochs)

    nssb = stage_vlines[-1]
    sebi = len(pkl.load(search_epochs[0].open("rb"))) // batch_size
    n_search_epochs = len(search_epochs)
    epoch_vlines = [nssb + i * sebi for i in range(1, n_search_epochs + 1)]

    n_valids = len(list(run_paths[0].glob("search_valid*.pkl")))
    vbi: float = json.load((run_paths[0] / "args.json").open("r"))["validate_every"]
    valid_xs = pd.Series([vbi * i for i in range(n_valids)])

    train_dfs = []
    valid_dfs = []

    for ri, pth in enumerate(run_paths):
        color = colors.pop()
        search_paths = natsorted(list(pth.glob("search_train*.pkl")))
        for p in search_paths:
            print(p.name)
        print("")
        search_trains = [pkl.load(p.open("rb")) for p in search_paths]
        model_trains = [
            pkl.load(p.open("rb"))
            for p in natsorted(list(pth.glob("model_train*.pkl")))
        ]
        valids = [
            pkl.load(p.open("rb"))
            for p in natsorted(list(pth.glob("search_valid*.pkl")))
        ]

        # fill model plots
        for mi, model_train in enumerate(model_trains):
            xs = model_train.index.values
            plot_single_ax(
                model_ax[0, 0],
                model_train["floss"],
                xs=xs,
                color=color,
            )
            if ri == 0 and mi == 0:
                add_vlines(
                    model_ax[0, 0], stage_vlines=stage_vlines, epoch_vlines=epoch_vlines
                )
            if "Levin" in run_name:
                plot_single_ax(
                    model_ax[1, 0],
                    model_train["facc"],
                    xs=xs,
                    color=color,
                )
                if ri == 0 and mi == 0:
                    add_vlines(
                        model_ax[1, 0],
                        stage_vlines=stage_vlines,
                        epoch_vlines=epoch_vlines,
                    )
            if "Bi" in run_name:
                plot_single_ax(
                    model_ax[0, 1],
                    model_train["bloss"],
                    xs=xs,
                    color=color,
                )
                if ri == 0 and mi == 0:
                    add_vlines(
                        model_ax[0, 1],
                        stage_vlines=stage_vlines,
                        epoch_vlines=epoch_vlines,
                    )
                if "Levin" in run_name:
                    plot_single_ax(
                        model_ax[1, 1],
                        model_train["bacc"],
                        xs=xs,
                        color=color,
                    )
                    if ri == 0 and mi == 0:
                        add_vlines(
                            model_ax[1, 1],
                            stage_vlines=stage_vlines,
                            epoch_vlines=epoch_vlines,
                        )

        # fill search plots
        seed_trains = []
        for si, search_train in enumerate(search_trains):
            seed_trains.append(search_train)

            # offset batch axis by stage
            if si == 0:
                start = 1
            else:
                start = sum(len(s) // batch_size for s in search_trains[:si]) + 1

            # plot solved
            solved = search_train["len"].fillna(0)
            solved[solved > 0] = 1
            assert len(solved) % batch_size == 0
            plot_single_ax(
                search_ax[0, 0],
                solved,
                aggr_size=batch_size,
                start=start,
                color=color,
            )
            if ri == 0 and si == 0:
                add_vlines(
                    search_ax[0, 0],
                    stage_vlines=stage_vlines,
                    epoch_vlines=epoch_vlines,
                )
            # plot Len
            plot_single_ax(
                search_ax[1, 0],
                search_train["len"],
                aggr_size=batch_size,
                start=start,
                color=color,
            )
            if ri == 0 and si == 0:
                add_vlines(
                    search_ax[1, 0],
                    stage_vlines=stage_vlines,
                    epoch_vlines=epoch_vlines,
                )

            # plot preds
            if "Levin" in run_name:
                preds = np.exp(-search_train["fpp"])
            else:
                preds = abs(search_train["fpp"] - search_train["len"])
            plot_single_ax(
                search_ax[2, 0],
                preds,
                aggr_size=batch_size,
                start=start,
                color=color,
            )
            if ri == 0 and si == 0:
                add_vlines(
                    search_ax[2, 0],
                    stage_vlines=stage_vlines,
                    epoch_vlines=epoch_vlines,
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
            if ri == 0 and si == 0:
                add_vlines(
                    search_ax[3, 0],
                    stage_vlines=stage_vlines,
                    epoch_vlines=epoch_vlines,
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
                if ri == 0 and si == 0:
                    add_vlines(
                        search_ax[0, 1],
                        stage_vlines=stage_vlines,
                        epoch_vlines=epoch_vlines,
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
                if ri == 0 and si == 0:
                    add_vlines(
                        search_ax[1, 1],
                        stage_vlines=stage_vlines,
                        epoch_vlines=epoch_vlines,
                    )

                # plot F/B preds
                if "Levin" in run_name:
                    fpreds = np.exp(-search_train["fpp"])
                    bpreds = np.exp(-search_train["bpp"])
                else:
                    fpreds = abs(search_train["fpp"] - search_train["len"])
                    bpreds = abs(search_train["bpp"] - search_train["len"])
                fb_pp = fpreds / bpreds
                plot_single_ax(
                    search_ax[2, 1],
                    fb_pp,
                    aggr_size=batch_size,
                    start=start,
                    color=color,
                    stage_vlines=stage_vlines,
                    epoch_vlines=epoch_vlines,
                )
                if ri == 0 and si == 0:
                    add_vlines(
                        search_ax[2, 1],
                        stage_vlines=stage_vlines,
                        epoch_vlines=epoch_vlines,
                    )

        seed_train_df = pd.concat(seed_trains, ignore_index=True)

        # fill valid plots
        # plot solved
        valid_solved = []
        valid_lens = []
        valid_exps = []
        valid_fb_lens = []
        valid_fb_exps = []
        seed_valids = []
        for search_valid in valids:
            seed_valids.append(search_valid)
            solved = search_valid["len"].fillna(0)
            solved[solved > 0] = 1
            assert len(solved) % batch_size == 0
            valid_solved.append(solved.mean())

            valid_lens.append(search_valid["len"].mean())

            total_exps = []
            if "Bi" in run_name:
                exps = search_valid["fexp"] + search_valid["bexp"]
                fb_exps = search_valid["fexp"] / search_valid["bexp"]
                fb_lens = search_valid["fg"] / search_valid["bg"]
                valid_fb_exps.append(fb_exps.mean())
                valid_fb_lens.append(fb_lens.mean())
            else:
                exps = search_valid["fexp"]
            total_exps.append(exps.sum())
            valid_exps.append(exps.mean())

        seed_valid_df = pd.concat(seed_valids, ignore_index=True)

        # plot solved
        plot_single_ax(
            valid_ax[0, 0],
            pd.Series(valid_solved),
            color=color,
            xs=valid_xs,
        )
        if ri == 0:
            add_vlines(
                valid_ax[0, 0], stage_vlines=stage_vlines, epoch_vlines=epoch_vlines
            )

        # plot Len
        plot_single_ax(
            valid_ax[1, 0],
            pd.Series(valid_lens),
            color=color,
            xs=valid_xs,
        )
        if ri == 0:
            add_vlines(
                valid_ax[1, 0], stage_vlines=stage_vlines, epoch_vlines=epoch_vlines
            )

        # plot exp
        total_exp = sum(valid_exps)
        plot_single_ax(
            valid_ax[2, 0],
            pd.Series(valid_exps),
            color=color,
            xs=valid_xs,
        )
        if ri == 0:
            add_vlines(
                valid_ax[2, 0], stage_vlines=stage_vlines, epoch_vlines=epoch_vlines
            )

        # plot fb exp and len
        if "Bi" in run_name:
            # plot F/B exp
            plot_single_ax(
                valid_ax[0, 1],
                pd.Series(valid_fb_exps),
                color=color,
                xs=valid_xs,
            )
            if ri == 0:
                add_vlines(
                    valid_ax[0, 1], stage_vlines=stage_vlines, epoch_vlines=epoch_vlines
                )

            # plot F/B len
            plot_single_ax(
                valid_ax[1, 1],
                pd.Series(valid_fb_lens),
                color=color,
                xs=valid_xs,
            )
            if ri == 0:
                add_vlines(
                    valid_ax[1, 1], stage_vlines=stage_vlines, epoch_vlines=epoch_vlines
                )

        train_dfs.append(seed_train_df)
        valid_dfs.append(seed_valid_df)

    plt.close("all")
    return model_fig, search_fig, valid_fig, train_dfs, valid_dfs


def group_and_plot(path, group_key_func, batch_size=40):
    runpaths = list(path.glob("*/"))
    runnames = []
    rundata = []
    runs = sorted(runpaths, key=group_key_func)
    for k, g in itertools.groupby(runs, group_key_func):
        runnames.append(k)
        rundata.append(list(g))

    figs = []
    all_train_dfs = []
    all_valid_dfs = []
    for run_name, run_paths in zip(runnames, rundata):
        print(f"Found {len(run_paths)} runs of {run_name}")
        model_fig, search_fig, valid_fig, train_dfs, valid_dfs = plot_same_axes(
            run_name, run_paths, batch_size
        )
        figs.append((run_name, model_fig, search_fig, valid_fig))
        all_train_dfs.append((run_name, train_dfs))
        all_valid_dfs.append((run_name, valid_dfs))

    return figs, all_train_dfs, all_valid_dfs


class PdfTemplate:
    def __init__(self, figsdir: Path, outfile: str, toc=True):
        self.figs = figsdir
        self.toc = toc
        self.outfile = outfile
        self.latex_cmds: list = []
        self.text: str = ""

    def render(self):
        self._pagebreak()
        for fig in sorted(self.figs.glob("*.png"), key=lambda x: x.stem):
            self._h1(fig.stem.replace("_", " "))
            self._img(str(fig))
            self._pagebreak()
        self.text = "\n\n".join(self.latex_cmds)

    def export(self):
        md_file = f"{self.outfile}.md"
        pdf_file = f"{self.outfile}.pdf"
        pandoc = ["pandoc", f"{md_file}", "-V geometry:margin=1cm", f"-o{pdf_file}"]
        with open(md_file, "w") as f:
            f.write(self.text)
        if self.toc:
            pandoc.append("--toc")
        subprocess.run(pandoc)

    def _pagebreak(self):
        self.latex_cmds.append("\pagebreak")

    def _h1(self, text):
        self.latex_cmds.append(f"# {text}")

    def _img(self, img):
        self.latex_cmds.append(f"![]({img}){{width=600px height=700px}}")


def main():
    time_window = 150
    batch_size = 40
    exp_name = "pancake16"
    levin_runs = Path(f"../final_runs/{exp_name}_levin/")
    astar_runs = Path(f"../final_runs/{exp_name}_astar/")
    figs_dir = Path(f"final_runs_{exp_name}")

    figs_dir.mkdir(exist_ok=True)

    levin_figs, levin_train_dfs, levin_valid_dfs = group_and_plot(
        levin_runs, levin_group_key, batch_size=batch_size
    )
    astar_figs, astar_train_dfs, astar_valid_dfs = group_and_plot(
        astar_runs, astar_group_key, batch_size=batch_size
    )
    figs = levin_figs + astar_figs

    # for run_name, model_fig, search_fig, valid_fig in figs:
    #     model_fig.savefig(figs_dir / f"{run_name.replace(' ', '_')}_model.png")
    #     search_fig.savefig(figs_dir / f"{run_name.replace(' ', '_')}_search.png")
    #     valid_fig.savefig(figs_dir / f"{run_name.replace(' ', '_')}_valid.png")

    # pdf = PdfTemplate(figs_dir, str(figs_dir))
    # pdf.render()
    # pdf.export()

    # plot solved vs exps
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 6))

    # tab_colors = mcolors.TABLEAU_COLORS
    # colors = [
    #     tab_colors["tab:blue"],
    #     tab_colors["tab:green"],
    #     tab_colors["tab:red"],
    #     tab_colors["tab:orange"],
    # ]
    colors = ["b", "c", "g", "m"]
    labels = []
    for run_name, dfs in itertools.chain(astar_train_dfs, levin_train_dfs):
        print(run_name)
        solveds = []
        cols = []
        curr_lens = len(dfs[0])
        end_curr = -1
        for i, df in enumerate(dfs):
            if end_curr < 0 and len(df) != curr_lens:
                end_curr = i
            df = df[["time", "len"]].copy()
            df["len"] = df["len"].apply(pd.notna)
            dfg = df.groupby(df.index // batch_size)
            dfs = dfg.aggregate({"time": "max", "len": "mean"})
            dfs["time"] = dfs["time"].cumsum()
            x = dfs.groupby(dfs["time"] // time_window)["len"].mean()
            x.index = x.index.map(lambda x: (x + 1) * time_window)
            solveds.append(x)
            cols.append(i)

        df = pd.concat(solveds, axis=1)
        df.columns = cols
        df["mean"] = df.mean(axis=1)
        df["min"] = df.min(axis=1)
        df["max"] = df.max(axis=1)

        # color = tab_colors.popitem()[1]
        color = colors.pop()
        ax.plot(df.index.values, df["mean"], label=run_name, color=color)
        plt.fill_between(
            np.array(df.index.values, dtype=np.float32),
            df["min"],
            df["max"],
            alpha=0.1,
            color=color,
            label=run_name,
        )
        labels.append(run_name)

    handler, labeler = ax.get_legend_handles_labels()
    it = iter(handler)
    hd = [(a, next(it)) for a in it]
    ax.legend(hd, labels)
    ax.set_xlabel("Time (s)", size=12)
    ax.set_ylabel("Solved", rotation=0, labelpad=30, size=12)
    ax.set_title("Solved vs. time")
    plt.savefig(figs_dir/ f"svt_{str(exp_name)}.pdf")
    # plt.show()


if __name__ == "__main__":
    main()
