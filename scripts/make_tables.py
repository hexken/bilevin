from collections import OrderedDict
from pathlib import Path
import pickle as pkl
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
import plotting.utils as putils


def reorder_agents(dom_data, agent_group=None):
    if agent_group is None:
        all = list(dom_data.items())
    all = [(a, d) for a, d in dom_data.items() if a in agent_group]
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
    print(f"1 len of solved ids: {len(solved_ids)}")

    r1_data["run"] = 1
    merged_data = [r1_data]

    for i, run in enumerate(runs[1:], start=2):
        rundata = data_func(run)
        i_solved_ids = set(rundata[rundata["len"].notna()]["id"].astype(int))
        solved_ids = solved_ids.intersection(i_solved_ids)
        print(f"{i} len of solved ids: {len(solved_ids)}")

        rundata["run"] = i
        merged_data.append(rundata)

    merged_data = pd.concat(merged_data)
    return merged_data, solved_ids


def compute_domain_stats(dom_data, common_ids=None, common_min: int = 100):
    s = ""
    describes = {}
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
        data["solved"] = data["len"].notna()
        if "bexp" in data:
            data["total_exp"] = data["fexp"] + data["bexp"]
        else:
            data["total_exp"] = data["fexp"]
        all_ids_stats = data.groupby("run").agg({"solved": "sum", "total_exp": "mean"})
        # print(solved)
        # print(solved.describe())
        stats = print_df.describe()
        stats["solved"] = all_ids_stats["solved"].describe()
        stats["total_exp"] = all_ids_stats["total_exp"].describe()
        # print(stats)
        describes[agent] = stats
        # if stats.loc["count"]["exp"] == 1:
        #     stats.loc["std"] = 0.0
        # elif stats.loc["count"]["exp"] == 0:
        #     continue

        # stats = stats.drop("count", axis=0)
        # s += str(stats)
        stats = stats.fillna(0)
        s += tabulate(
            stats,
            headers="keys",
            showindex=True,
            floatfmt=".3f",
        )
        s += "\n\n"
    return s, describes


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python make_tables.py <indir> <outdir>")
        sys.exit(1)

    inq = Path(sys.argv[1])

    outdir = Path(sys.argv[2])
    outdir.mkdir(exist_ok=True, parents=True)

    common_min = 100

    for dom in tqdm(putils.allowable_domains):
        print(f"Processing {dom}")
        base_string = f"Domain {dom}\n"
        dom_data = pkl.load((inq / f"{dom}_trim.pkl").open("rb"))

        agent_group = getattr(putils, dom)
        print(f"Processing agent group: {agent_group}")
        dom_data = reorder_agents(dom_data, agent_group)

        # find agent solved ids
        agents_data = OrderedDict()
        print(list(dom_data.keys()))
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
            common_string = (
                base_string
                + f"Skipping {dom} due to insufficient common solved problems"
            )
        else:
            common_file = (outdir / f"{dom}_common.txt").open("w")
            print(f"writing common")
            all_agents = set(dom_data.keys())
            common_string = base_string + f"Excluding agents: {*exclude_agents,}\n\n"
            common_agents = {
                a: d for a, d in agents_data.items() if a not in exclude_agents
            }
            s, d = compute_domain_stats(common_agents, common_ids)
            common_string += s
            pkl.dump(d, (outdir / f"{dom}_common_stats.pkl").open("wb"))
            common_file.write(common_string)

        solved_file = (outdir / f"{dom}_solved.txt").open("w")
        print(f"writing solved")
        s, d = compute_domain_stats(agents_data)
        solved_string = base_string + s
        solved_file.write(solved_string)
        pkl.dump(d, (outdir / f"{dom}_solved_stats.pkl").open("wb"))

        all_file = (outdir / f"{dom}_all.txt").open("w")
        print(f"writing all")
        s, d = compute_domain_stats(agents_data, all_ids)
        pkl.dump(d, (outdir / f"{dom}_all_stats.pkl").open("wb"))
        all_string = base_string + s
        all_file.write(all_string)
