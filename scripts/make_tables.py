from collections import OrderedDict
from pathlib import Path
import pickle as pkl
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate


def reorder_agents(dom_data):
    new_dom_data = OrderedDict()
    order = ("PHS", "BiPHS", "Levin", "BiLevin", "AStar", "BiAStar")
    for agent in order:
        assert agent in dom_data
        new_dom_data[agent] = dom_data[agent]
    return new_dom_data


def get_common_ids_agent(data: list[pd.DataFrame], mode: str):
    """
    Get the common problems that a single agents solved in the last epoch of a domain, over all
    seeds
    """

    if mode == "test":
        data_func = lambda x: x.copy()
    elif mode == "train":
        data_func = lambda x: x[x["epoch"] == 10].copy()
    elif mode == "valid":
        data_func = lambda x: x.iloc[-1000:].copy()

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


def get_common_ids_domain(dom_data: dict, mode: str):
    """
    Get the common problems that all agents solved in the last epoch of a domain, over all seeds,
    and the common ids accross all agents
    """
    ret = OrderedDict()
    dom_data_l = list(dom_data.items())
    if mode == "test":
        data_func = lambda x: x
    elif mode == "train":
        data_func = lambda x: x["search"]
    elif mode == "valid":
        data_func = lambda x: x["valid"]

    agent, adata = dom_data_l[0]
    data, ids = get_common_ids_agent(data_func(adata.data), mode=mode)
    all_ids = ids.copy()
    ret[agent] = {}
    ret[agent]["ids"] = ids
    ret[agent]["data"] = data

    for agent, adata in dom_data_l[1:]:
        ret[agent] = {}
        ret[agent]["ids"] = ids
        ret[agent]["data"] = ret
        all_ids = all_ids.intersection(ids)

    ret["all"] = all_ids
    return ret


def compute_domain_stats(dom_data, mode: str, common_min: int = 100):
    dom_data = get_common_ids_domain(dom_data, mode=mode)

    all_ids = dom_data.pop("all")
    for agent, adata in dom_data.items():
        if len(adata["ids"]) < common_min:
            print(f"Skipping {agent} due to insufficient common solved problems")
            continue

        data = adata["data"]
        policy = False
        heuristic = False
        bidir = False
        print(data)
        print_df = {"len": data["len"]}
        # lens = []
        # exps = []
        print(f"{agent}")
        print(f"Common solved: {len(adata['ids'])}")
        ids = adata["ids"]
        mask = data["id"].isin(ids)
        id_data = data.loc[mask]
        # print(f"{dom} {agent} {len(epoch_search_probs)}")
        group_id_problems = id_data.groupby("id").agg(["mean", "std"])

        if "bexp" in data:
            bidir = True
            data["exp"] = id_data["fexp"] + id_data["bexp"]
            print_df["fb_len"] = id_data["fg"] / (id_data["fg"] + id_data["bg"])
            # fb_lens = []
            # fb_exps = []
        else:
            print_df["exp"] = id_data["fexp"]

        if "fap" in data:
            policy = True
            fap = []
            if bidir:
                bap = []
                print_df["fb_ap"] = id_data["fap"] / (id_data["fap"] + id_data["bap"])
                fb_ap = []

        if "fhe" in data:
            heuristic = True
            fhe = []
            if bidir:
                bhe = []
                print_df["fb_he"] = id_data["fhe"] / (id_data["fhe"] + id_data["bhe"])
                fb_he = []

        tabulate(
            group_id_problems,
            tablefmt="psql",
            floatfmt=".2f",
        )
        # print(
        #     f"{dom} {key} {agent} \n\texps {np.mean(exps):.3f} +- {np.std(exps):.3f} \n\tlens {np.mean(lens):.3f} +- {np.std(lens):.3f}"
        # )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python common_problems.py <indir> <train|valid|test>")
        sys.exit(1)

    inq = Path(sys.argv[1])
    doms = ("tri4", "tri5", "col4", "col5", "stp4", "stp5")
    # agents = {
    #     "phs": ("_PHS", "_BiPHS"),
    #     "levin": ("_Levin", "_BiLevin"),
    #     "astar": ("_AStar", "_BiAStar"),
    # }
    mode = sys.argv[2]
    if mode not in ("train", "valid", "test"):
        print("Mode must be one of train, valid, test")
        sys.exit(1)

    for dom in doms:
        # compute common problems for this domain
        dom_data = pkl.load((inq / f"{dom}.pkl").open("rb"))
        dom_data = reorder_agents(dom_data)
        compute_domain_stats(dom_data, mode=mode)
