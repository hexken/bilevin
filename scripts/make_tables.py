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


def get_common_ids_agent(data: list[pd.DataFrame], mode: str, common_min: int = 100):
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


def get_common_ids_domain(dom_data: dict, mode: str, common_min: int = 100):
    """
    Get the common problems that all agents solved in the last epoch of a domain, over all seeds,
    and the common ids accross all agents. Skip agents that have less than `common_min` common ids
    solved
    """
    if mode == "test":
        data_func = lambda x: x
    elif mode == "train":
        data_func = lambda x: x["search"]
    elif mode == "valid":
        data_func = lambda x: x["valid"]

    ret = OrderedDict()
    max_ids = set()
    for agent, adata in dom_data.items():
        data, ids = get_common_ids_agent(data_func(adata.data), mode=mode)

        if len(ids) < common_min:
            print(f"skipping {agent} due to insufficient common solved problems")
            continue

        if len(ids) > len(max_ids):
            max_ids = ids

        ret[agent] = {"ids": ids, "data": data}

    common_ids = max_ids.intersection(*[x["ids"] for x in ret.values()])
    ret["all"] = common_ids

    return ret


def compute_domain_stats(dom_data, mode: str, common_min: int = 100):
    dom_common_data = get_common_ids_domain(dom_data, mode=mode)

    common_ids = dom_common_data.pop("all")
    if len(common_ids) < common_min:
        print(f"Skipping domain {dom} due to insufficient common solved problems")
        return

    print(f"Domain {dom} common solved: {len(common_ids)}")
    print("Using agents:")
    print(*dom_common_data.keys())
    print()
    for agent, adata in dom_common_data.items():

        data = adata["data"]
        print(f"{agent}")
        print(f"Common solved: {len(adata['ids'])}")
        mask = data["id"].isin(common_ids)
        id_data = data.loc[mask]
        print_df = {"id": id_data["id"], "len": id_data["len"]}

        if "bexp" in data:
            print_df["fb_len"] = id_data["fg"] / (id_data["fg"] + id_data["bg"])
            print_df["exp"] = id_data["fexp"] + id_data["bexp"]
            print_df["fb_exp"] = id_data["fexp"] / (id_data["fexp"] + id_data["bexp"])
        else:
            print_df["exp"] = id_data["fexp"]

        if "fap" in data:
            print_df["fap"] = id_data["fap"]
            if "bap" in data:
                print_df["fb_ap"] = id_data["fap"] / (id_data["fap"] + id_data["bap"])

        if "fhe" in data:
            print_df["fhe"] = id_data["fhe"]
            if "bhe" in data:
                print_df["fb_he"] = id_data["fhe"] / (id_data["fhe"] + id_data["bhe"])

        print_df = pd.DataFrame(print_df)
        print_df = print_df.groupby("id").mean()
        # stats = print_df.describe().map(lambda x: "{0:.3f}".format(x))
        # print(stats)
        stats = print_df.describe()
        print(
            tabulate(
                stats,
                headers="keys",
                showindex=True,
                floatfmt=".3f",
            )
        )
        print("\n\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python common_problems.py <indir> <train|valid|test>")
        sys.exit(1)

    inq = Path(sys.argv[1])
    doms = ("tri4", "tri5", "col4", "col5", "stp4", "stp5")
    mode = sys.argv[2]
    if mode not in ("train", "valid", "test"):
        print("Mode must be one of train, valid, test")
        sys.exit(1)

    for dom in doms:
        dom_data = pkl.load((inq / f"{dom}.pkl").open("rb"))
        dom_data = reorder_agents(dom_data)
        compute_domain_stats(dom_data, mode=mode)
