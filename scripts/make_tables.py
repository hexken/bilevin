from collections import OrderedDict
from pathlib import Path
import pickle as pkl
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
from plotting.utils import allowable_domains


def reorder_agents(dom_data):
    all = [(a, d) for a, d in dom_data.items() if "_m" not in a and "w1" not in a]
    all = sorted(all, key=lambda x: len(x[0]))
    all = sorted(all, key=lambda x: x[0].split("Bi")[-1])
    return OrderedDict(all)


def get_common_ids_agent(runs: list[dict], common_min: int = 100):
    """
    Get the common problems that a single agents solved during testing of a domain, over all
    seeds. Merge results into one DF, keyed by run #
    """

    data_func = lambda x: x["test"].copy()

    r1_data = data_func(runs[0])
    solved_ids = set(r1_data[r1_data["len"].notna()]["id"].astype(int))

    r1_data["run"] = 1
    merged_data = [r1_data]

    for i, run in enumerate(runs[1:], start=2):
        rundata = data_func(run)
        solved_ids = set(rundata[rundata["len"].notna()]["id"].astype(int))
        solved_ids = solved_ids.intersection(solved_ids)

        rundata["run"] = i
        merged_data.append(rundata)

    merged_data = pd.concat(merged_data)
    return merged_data, solved_ids


def compute_domain_stats(dom_data, common_ids=None, common_min: int = 100):
    s = ""
    for agent, (data, ids) in dom_data.items():
        s += f"{agent}\n"
        if common_ids is not None:
            mask = data["id"].isin(common_ids)
        else:
            mask = data["id"].isin(ids)
        id_data = data.loc[mask]
        print_df = {"id": id_data["id"], "len": id_data["len"], "time": id_data["time"]}

        if "bexp" in data:
            print_df["fb_len"] = id_data["fg"] / (id_data["fg"] + id_data["bg"])
            print_df["exp"] = id_data["fexp"] + id_data["bexp"]
            print_df["fb_exp_b"] = id_data["fexp"] / (id_data["fexp"] + id_data["bexp"])
            print_df["fb_exp_u"] = id_data["fexp"] / id_data["bexp"]
        else:
            print_df["exp"] = id_data["fexp"]

        if "fap" in data:
            print_df["fap"] = id_data["fap"]
            if "bap" in data:
                print_df["fb_ap_b"] = id_data["fap"] / (id_data["fap"] + id_data["bap"])
                print_df["fb_ap_u"] = id_data["fap"] / id_data["bap"]

        if "fhe" in data:
            print_df["fhe"] = id_data["fhe"]
            if "bhe" in data:
                print_df["fb_he_b"] = id_data["fhe"] / (id_data["fhe"] + id_data["bhe"])
                print_df["fb_he_u"] = id_data["fhe"] / id_data["bhe"]

        print_df = pd.DataFrame(print_df)
        # print(print_df.head())
        print_df = print_df.groupby("id").mean()
        # stats = print_df.describe().map(lambda x: "{0:.3f}".format(x))
        # print(stats)
        stats = print_df.describe()
        if stats.loc["count"]["exp"] == 1:
            stats.loc["std"] = 0.0
        elif stats.loc["count"]["exp"] == 0:
            continue

        # stats = stats.drop("count", axis=0)
        # print(stats)
        s += tabulate(
            stats,
            headers="keys",
            showindex=True,
            floatfmt=".3f",
        )
        s += "\n\n"
    return s


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python make_tables.py <indir> <outdir>")
        sys.exit(1)

    inq = Path(sys.argv[1])

    outdir = Path(sys.argv[2])
    outdir.mkdir(exist_ok=True, parents=True)

    common_min = 100

    for dom in tqdm(allowable_domains):
        print(f"Processing {dom}")
        s = f"Domain {dom}\n"

        dom_data = pkl.load((inq / f"{dom}.pkl").open("rb"))
        dom_data = reorder_agents(dom_data)

        # find agent solved ids
        agents_data = OrderedDict()
        all_ids = set(list(dom_data.values())[0][0]["test"]["id"].astype(int))
        # print(all_ids)
        for agent, runs in dom_data.items():
            (
                agent_merged_data,
                agent_solved_ids,
            ) = get_common_ids_agent(runs)
            agents_data[agent] = (agent_merged_data, agent_solved_ids)

        common_ids = all_ids
        exclude_agents = set()
        for agent, (adata, aids) in agents_data.items():

            if len(aids) < common_min:
                exclude_agents.add(agent)
                print(f"Excluding {agent} due to insufficient solved problems")
                continue
            else:
                common_ids = common_ids.intersection(aids)

        print(f"domain {dom} has {len(common_ids)} common ids")
        if len(common_ids) < common_min:
            s += f"Skipping {dom} due to insufficient common solved problems"
        else:
            common_file = (outdir / f"{dom}_common.txt").open("w")
            print(f"writing common")
            all_agents = set(dom_data.keys())
            s += f"Excluding agents: {*exclude_agents,}\n\n"
            common_agents = {
                a: d for a, d in agents_data.items() if a not in exclude_agents
            }
            s += compute_domain_stats(common_agents, common_ids)
            # s = compute_common_domain_stats(dom, dom_data)
            common_file.write(s)
            # print(s)

        solved_file = (outdir / f"{dom}_solved.txt").open("w")
        print(f"writing solved")
        s += compute_domain_stats(agents_data)
        solved_file.write(s)
        # print(s)

        all_file = (outdir / f"{dom}_all.txt").open("w")
        print(f"writing all")
        s += compute_domain_stats(agents_data)
        all_file.write(s)
