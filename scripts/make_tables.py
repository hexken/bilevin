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


def get_common_train(data: list[pd.DataFrame]):
    """
    Get the common problems that a single agents solved in the last epoch of a domain, over all
    seeds
    """
    r1_data = data[0]
    r1_data = r1_data[r1_data["epoch"] == 10]
    ids = set(r1_data[r1_data["len"] > 0]["id"].astype(int))

    r1_data["run"] = 1
    merged_data = [r1_data]

    # get common ids
    for i, run in enumerate(data[1:], start=2):
        rundata = run[run["epoch"] == 10]
        solved_ids = rundata[rundata["len"] > 0]["id"].astype(int)
        ids = ids.intersection(solved_ids)
        rundata["run"] = i
        merged_data.append(rundata)

    merged_data = pd.concat(merged_data)
    return merged_data, ids


def get_common(data: list[pd.DataFrame], mode=str):
    """
    Get the common problems that a single agents solved in the last epoch of a domain, over all
    seeds
    """

    if mode == "test":
        data_func = lambda x: x
    elif mode == "train":
        data_func = lambda x: x[x["epoch"] == 10]
    elif mode == "valid":
        data_func = lambda x: x.iloc[-1000:]

    r1_data = data_func(data[0])
    ids = set(r1_data[r1_data["len"] > 0]["id"].astype(int))

    r1_data["run"] = 1
    merged_data = [r1_data]

    for i, run in enumerate(data[1:], start=2):
        rundata = data_func(run)
        solved_ids = set(rundata[rundata["len"] > 0]["id"].astype(int))
        ids = ids.intersection(solved_ids)

        rundata["run"] = i
        merged_data.append(rundata)

    merged_data = pd.concat(merged_data)
    return merged_data, ids


def get_common_valid(data: list[pd.DataFrame]):
    """
    Get the common problems that a single agents solved in the last epoch of a domain, over all
    seeds
    """
    r1_data = data[0].iloc[-1000:]
    ids = set(r1_data[r1_data["solved"] == True]["id"].astype(int))

    r1_data["run"] = 1
    merged_data = [r1_data]

    for i, run in enumerate(data[1:], start=2):
        rundata = run.iloc[-1000:]
        solved_ids = rundata[rundata["solved"] == True]["id"].astype(int)
        rundata["run"] = i
        merged_data.append(rundata)
        ids = ids.intersection(solved_ids)

    merged_data = pd.concat(merged_data)
    return merged_data, ids


def get_common_test(data: list[pd.DataFrame]):
    """
    Get the common test problems that a single agent solved over all trained models
    """
    r1_data = data[0]
    ids = set(r1_data[r1_data["len"] > 0]["id"].astype(int))

    r1_data["run"] = 1
    merged_data = [r1_data]

    for i, r1_data in enumerate(data, start=2):
        solved_ids = r1_data[r1_data["len"] > 0]["id"].astype(int)
        ids = ids.intersection(solved_ids)
        r1_data["run"] = i
        merged_data.append(r1_data)

    merged_data = pd.concat(merged_data)
    return merged_data, ids


def get_common_domain(dom_data, mode=str):
    """
    Get the common problems that all agents solved in the last epoch of a domain, over all seeds,
    and the common ids accross all agents
    """
    ret = OrderedDict()
    dom_data = list(dom_data.items())
    if mode == "test":
        data_func = lambda x: x
    elif mode == "train":
        data_func = lambda x: x["train"]
    elif mode == "valid":
        data_func = lambda x: x["valid"]

    agent, adata = dom_data[0]
    data, ids = get_common(data_func(adata.data), mode=mode)
    all_ids = ids.copy()
    ret[agent]["ids"] = ids
    ret[agent]["data"] = data

    for agent, adata in dom_data[1:]:
        ret[agent]["ids"] = ids
        ret[agent]["data"] = ret
        all_ids = all_ids.intersection(ids)

    ret["all"] = all_ids
    return ret


def compute_domain_stats(dom_data, common_min: int = 100, mode=str):
    dom_data = get_common_domain(dom_data, mode=mode)

    for agent, adata in dom_data.items():
        data = adata["data"]
        lens = []
        exps = []
        print(f"{agent}")
        print(f"Common solved: {len(adata['ids'])}")

        if len(adata["ids"]) < common_min:
            print(f"Skipping {agent} due to insufficient common solved problems")
            continue

        if "bexp" in adata["data"]:
            bidir = True
            fb_lens = []
            fb_exps = []

        if "fap" in adata["data"]:
            policy = True
            fap = []
            if bidir:
                bap = []
                fb_ap = []

        if "fhe" in adata["data"]:
            heuristic = True
            fhe = []
            if bidir:
                bhe = []
                fb_he = []

        ids = adata["ids"]
        mask = adata["id"].isin(ids)
        id_problems = data.loc[mask]
        # print(f"{dom} {agent} {len(epoch_search_probs)}")
        group_id_problems = id_problems.groupby("id").agg(kk)
        lens.append(group_runs["len"].mean())
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
        print("Usage: python common_problems.py <indir> <train|valid|test>")
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
                    valid_epoch_data[valid_epoch_data["solved"] == True]["id"].astype(
                        int
                    )
                )
                common_valid_ids = common_valid_ids.intersection(valid_solved_ids)
        common_ids[dom] = {"search": common_search_ids, "valid": common_valid_ids}
        print(f"{dom} search {len(common_search_ids)} valid {len(common_valid_ids)}")

        # compute statistics for common problems
        search_ids = common_ids[dom]["search"]
        valid_ids = common_ids[dom]["valid"]
        for key in keys:
            compute_train_stats(key, dom, dom_data, common_ids[dom][key])
