from collections import OrderedDict
from pathlib import Path
import pickle as pkl
import sys

import numpy as np


def reorder_agents(dom_data):
    new_dom_data = OrderedDict()
    order = ("PHS", "BiPHS", "Levin", "BiLevin", "AStar", "BiAStar")
    for agent in order:
        assert agent in dom_data
        new_dom_data[agent] = dom_data[agent]
    return new_dom_data


def compute_stats(dom, dom_data, ids):
    for agent, adata in dom_data.items():
        if dom == "col5" and agent == "AStar":
            continue
        if dom == "stp5" and agent == "BiPHS":
            continue
        keys = ["search", "valid"]
        lens = []
        exps = []
        if "Bi" in agent:
            fb_lens = []
            fb_exps = []
        for key in keys:
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
                f"{dom} {key} {agent} \n\texps {np.mean(exps)} \n\tlens {np.mean(lens)}"
            )
            if "Bi" in agent:
                print(f"\tfb_exps {np.mean(fb_exps)} \n\tfb_lens {np.mean(fb_lens)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python common_problems.py <indir> <outfile>")
        sys.exit(1)

    indir = Path(sys.argv[1])
    doms = ["col4", "col5", "tri4", "tri5", "stp4", "stp5"]
    common_ids = dict()
    for dom in doms:
        # compute common problems for this domain
        dom_data = pkl.load(Path(f"{indir}/{dom}.pkl").open("rb"))
        dom_data = reorder_agents(dom_data)
        common_valid_ids = set(i for i in range(0, 1000))
        if "stp" in dom:
            common_search_ids = set(i for i in range(50000, 100000))
        else:
            common_search_ids = set(i for i in range(0, 50000))

        for agent, adata in dom_data.items():
            # because astar sucked on col5
            if dom == "col5" and agent == "AStar":
                continue
            if dom == "stp5" and agent == "BiPHS":
                continue
            for run in adata.data["search"]:
                search_epoch_data = run[run["epoch"] == 10]
                search_solved_ids = set(
                    search_epoch_data[search_epoch_data["len"] > 0]["id"].astype(int)
                )
                common_search_ids = common_search_ids.intersection(search_solved_ids)
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

        compute_stats(dom, dom_data, search_ids)
        compute_stats(dom, dom_data, valid_ids)

    pkl.dump(common_ids, Path(sys.argv[2]).open("wb"))
