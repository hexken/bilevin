from collections import OrderedDict
from pathlib import Path
import pickle as pkl
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

ALL_AGENTS = ("PHS", "BiPHS", "Levin", "BiLevin", "AStar", "BiAStar")


def reorder_agents(dom_data):
    new_dom_data = OrderedDict()
    for agent in ALL_AGENTS:
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
    all_ids = set(r1_data["id"].astype(int))

    r1_data["run"] = 1
    merged_data = [r1_data]

    for i, run in enumerate(data[1:], start=2):
        rundata = data_func(run)
        solved_ids = set(rundata[rundata["len"] > 0]["id"].astype(int))
        ids = ids.intersection(solved_ids)

        rundata["run"] = i
        merged_data.append(rundata)

    merged_data = pd.concat(merged_data)
    return merged_data, ids, all_ids


def get_common_ids_domain(dom_data: dict, mode: str, common_min: int = 100):
    """
    Get the common problems that all agents solved in the last epoch of a domain, over all seeds,
    and the common ids accross all agents. Skip agents that have less than `common_min` common ids
    solved
    ret[agent][ids] are the common ids solved by the agent
    common_ids are the common ids solved by all agents
    all_ids are all ids of the final epoch or whatever problems are being considered
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
        data, ids, all_ids = get_common_ids_agent(data_func(adata.data), mode=mode)

        if len(ids) < common_min:
            print(f"skipping {agent} due to insufficient common solved problems")
            continue

        if len(ids) > len(max_ids):
            max_ids = ids

        ret[agent] = {"ids": ids, "data": data}

    common_ids = max_ids.intersection(*[x["ids"] for x in ret.values()])

    return ret, common_ids, all_ids


def compute_common_domain_stats(dom, dom_data, mode: str, common_min: int = 100):
    s = f"Domain {dom}\n"
    dom_common_data, common_ids, all_ids = get_common_ids_domain(dom_data, mode=mode)

    if len(common_ids) < common_min:
        s += f"Skipping due to insufficient common solved problems"
        return s

    # Update to use common ids for all agents
    for agent in dom_common_data:
        dom_common_data[agent]["ids"] = common_ids

    exluded = set(ALL_AGENTS) - set(dom_common_data.keys())
    s += f"Excluding agents: {*exluded,}\n\n"
    s += compute_domain_stats(dom_common_data, mode=mode)
    return s


def compute_solved_domain_stats(dom, dom_data, mode: str, common_min: int = 100):
    s = f"Domain {dom}\n"
    dom_common_data, common_ids, all_ids = get_common_ids_domain(dom_data, mode=mode)
    s += compute_domain_stats(dom_common_data, mode=mode, common_min=common_min)
    return s


def compute_unsolved_domain_stats(dom, dom_data, mode: str, common_min: int = 100):
    s = f"Domain {dom}\n"
    dom_common_data, common_ids, all_ids = get_common_ids_domain(dom_data, mode=mode)

    # Update to use unsolved ids
    for agent in dom_common_data:
        dom_common_data[agent]["ids"] = all_ids - dom_common_data[agent]["ids"]

    s += compute_domain_stats(dom_common_data, mode=mode)
    return s


def compute_all_domain_stats(dom, dom_data, mode: str, common_min: int = 100):
    s = f"Domain {dom}\n"
    dom_common_data, common_ids, all_ids = get_common_ids_domain(dom_data, mode=mode)

    # Update to use all ids
    for agent in dom_common_data:
        dom_common_data[agent]["ids"] = all_ids

    s += compute_domain_stats(dom_common_data, mode=mode, common_min=common_min)
    return s


def compute_domain_stats(dom_data, mode: str, common_min: int = 100):
    s = ""
    for agent, adata in dom_data.items():
        data = adata["data"]
        ids = adata["ids"]
        s += f"{agent}\n"
        mask = data["id"].isin(ids)
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
        if stats.loc["count"]["exp"] == 1:
            stats.loc["std"] = 0.0
        elif stats.loc["count"]["exp"] == 0:
            continue

        # stats = stats.drop("count", axis=0)
        print(stats)
        s += tabulate(
            stats,
            headers="keys",
            showindex=True,
            floatfmt=".3f",
        )
        s += "\n\n"
    return s


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python common_problems.py <indir> <train|valid|test> <outdir>")
        sys.exit(1)

    inq = Path(sys.argv[1])
    doms = ("tri4", "tri5", "col4", "col5", "stp4", "stp5")
    mode = sys.argv[2]
    if mode not in ("train", "valid", "test"):
        print("Mode must be one of train, valid, test")
        sys.exit(1)

    outdir = Path(sys.argv[3])
    outdir.mkdir(exist_ok=True, parents=True)

    for dom in tqdm(doms):
        print(f"Processing {dom}")

        dom_data = pkl.load((inq / f"{dom}.pkl").open("rb"))
        dom_data = reorder_agents(dom_data)

        common_file = (outdir / f"{dom}_{mode}_common.txt").open("w")
        print(f"writing common")
        s = compute_common_domain_stats(dom, dom_data, mode=mode)
        common_file.write(s)
        # print(s)

        solved_file = (outdir / f"{dom}_{mode}_solved.txt").open("w")
        print(f"writing solved")
        s = compute_solved_domain_stats(dom, dom_data, mode=mode)
        solved_file.write(s)
        # print(s)

        unsolved_file = (outdir / f"{dom}_{mode}_unsolved.txt").open("w")
        print(f"writing unsolved")
        s = compute_unsolved_domain_stats(dom, dom_data, mode=mode)
        unsolved_file.write(s)
        # print(s)

        all_file = (outdir / f"{dom}_{mode}_all.txt").open("w")
        print(f"writing all")
        s = compute_all_domain_stats(dom, dom_data, mode=mode)
        all_file.write(s)
        # print(s)
