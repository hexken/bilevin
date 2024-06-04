from collections import OrderedDict
from pathlib import Path
import pickle as pkl
import sys

import numpy as np
import pandas as pd


def reorder_agents(dom_data):
    new_dom_data = OrderedDict()
    order = ("PHS", "BiPHS", "Levin", "BiLevin", "AStar", "BiAStar")
    for agent in order:
        assert agent in dom_data
        new_dom_data[agent] = dom_data[agent]
    return new_dom_data


def get_common_train_best_expanded(data: pd.DataFrame):
    run1 = data[0]
    ids = set(run1[run1["epoch"] == 10]["id"].astype(int))
    for run in data[1:]:
        rundata = run[run["epoch"] == 10]
        solved = rundata[rundata["len"] > 0]["id"].astype(int)
        ids = ids.intersection(solved)
    return ids


def get_common_valid_best_expanded(data: pd.DataFrame):
    run1 = data[0]
    ids = set(run1.iloc[-1000:]["id"].astype(int))
    for run in data[1:]:
        rundata = run.iloc[-1000:]
        solved = rundata[rundata["solved"] == True]["id"].astype(int)
        ids = ids.intersection(solved)
    return ids

def get_common_train_latest(data: pd.DataFrame):
    run1 = data[0]
    ids = set(run1[run1["epoch"] == 10]["id"].astype(int))
    for run in data[1:]:
        rundata = run[run["epoch"] == 10]
        solved = rundata[rundata["len"] > 0]["id"].astype(int)
        ids = ids.intersection(solved)
    return ids


def get_common_valid_latest(data: pd.DataFrame):
    run1 = data[0]
    ids = set(run1.iloc[-1000:]["id"].astype(int))
    for run in data[1:]:
        rundata = run.iloc[-1000:]
        solved = rundata[rundata["solved"] == True]["id"].astype(int)
        ids = ids.intersection(solved)
    return ids


def get_common_test(agent_dir, agent, model_suffix):
    for run in agent_dir.glob("*/"):
        if agent in run.name:
            test_dirs = list(run.glob(f"*test_model_{model_suffix}*"))


def get_common_domain(dom_pkl, agents_common_problems):
    dom_data = pkl.load(dom_pkl.open("rb"))
    dom_data = reorder_agents(dom_data)


def compute_train_stats(key, dom, dom_data, ids, ignore=None):
    for agent, adata in dom_data.items():
        lens = []
        exps = []
        if ignore and dom in ignore and agent in ignore[dom]:
            continue
        if "Bi" in agent:
            fb_lens = []
            fb_exps = []
        for run in adata.data[key]:
            if key == "search":
                epoch_data = run[run["epoch"] == 10]
            else:
                epoch_data = run.iloc[-1000:]
            mask = epoch_data["id"].isin(ids)
            epoch_probs = epoch_data.loc[mask]
            # print(f"{dom} {agent} {len(epoch_search_probs)}")
            lens.append(epoch_probs["len"].mean())
            if "Bi" in agent:
                fexps = epoch_probs["fexp"]
                bexps = epoch_probs["bexp"]
                fbexps = fexps / (fexps + bexps)
                exps.append((fexps + bexps).mean())
                fb_exps.append(fbexps.mean())

                flens = epoch_probs["fg"]
                blens = epoch_probs["bg"]
                lens.append((flens + blens).mean())
                fblens = flens / (flens + blens)
                fb_lens.append(fblens.mean())
            else:
                exps.append(epoch_probs["exp"].mean())
                lens.append(epoch_probs["fg"].mean())
        print(
            f"{dom} {key} {agent} \n\texps {np.mean(exps):.3f} +- {np.std(exps):.3f} \n\tlens {np.mean(lens):.3f} +- {np.std(lens):.3f}"
        )
        if "Bi" in agent:
            print(
                f"\tfb_exps {np.mean(fb_exps):.3f} +- {np.std(fb_exps):.3f} \n\tfb_lens {np.mean(fb_lens):.3f} +- {np.std(fb_lens):.3f}"
            )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python common_problems.py <indir> <train|test>")
        sys.exit(1)

    inq = Path(sys.argv[1])
    test = sys.argv[2] == "test"
    doms = ("tri4", "tri5", "col4", "col5", "stp4", "stp5")
    agents = {
        "phs": ("_PHS", "_BiPHS"),
        "levin": ("_Levin", "_BiLevin"),
        "astar": ("_AStar", "_BiAStar"),
    }
    for dom in doms:
        # compute common problems for this domain
        dom_data = pkl.load((inq / f"{dom}.pkl").open("rb"))
        dom_data = reorder_agents(dom_data)

        for agent, adata in dom_data.items():
            # because astar sucked on col5
            if dom == "col5" and agent == "AStar":
                continue
            if dom == "stp5" and agent == "BiPHS":
                continue
            for run in adata.data["valid"]:
                valid_epoch_data = run.iloc[-1000:]
                valid_solved_ids = set(
                    valid_epoch_data[valid_epoch_data["solved"] == True][
                        "id"
                    ].astype(int)
                )
                common_valid_ids = common_valid_ids.intersection(valid_solved_ids)
        common_ids[dom] = {"search": common_search_ids, "valid": common_valid_ids}
        print(
            f"{dom} search {len(common_search_ids)} valid {len(common_valid_ids)}"
        )

        # compute statistics for common problems
        search_ids = common_ids[dom]["search"]
        valid_ids = common_ids[dom]["valid"]
        for key in keys:
            compute_train_stats(key, dom, dom_data, common_ids[dom][key])
