import operator
from pathlib import Path
import pickle as pkl

import pandas as pd

doms = [
    (r"\wittri\ 4", "tri4"),
    (r"\wittri\ 5", "tri5"),
    (r"\witcol\ 4", "col4"),
    (r"\witcol\ 5", "col5"),
    (r"\stp\ 4", "stp4"),
    (r"\stp\ 5", "stp5"),
    (r"\pancake\ 10", "pancake10"),
    (r"\pancake\ 12", "pancake12"),
]

data_rows = [("E", "exp", "<"), ("L", "len", "<")]
agent_columns = ("AStar", "BiAStar", "Levin", "BiLevin", "PHS", "BiPHS")
agent_types = ("AStar", "Levin", "PHS")


def compare(a, b, op):
    ops = {
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
        ">=": operator.ge,
        ">": operator.gt,
    }
    return ops[op](a, b)


def create_common_table(data: dict[str, pd.DataFrame], name: str) -> str:
    latex_table = [
        r"\begin{table}[htbp]",
        r"\footnotesize",
        r"\centering",
        r"\begin{tabularx}{\columnwidth}{cc|cc|cc|cc}",
        r"\phantom{} & \phantom{} & \mastar & \mbiastar & \lts & \bilts & \phs & \biphs \\",
        r"\cmidrule{3-8}",
    ]

    for ldom, pdom in doms:
        dom_data = data[pdom]
        # get proper keys for each agent
        matching_keys = [
            next((key for key in dom_data if key.startswith(prefix)), None)
            for prefix in agent_columns
        ]
        # group proper keys by agent type
        agents_data = {at: {} for at in agent_types}
        group_best_agents = {
            at: {prow: set() for _, prow, _ in data_rows} for at in agent_types
        }
        overall_best_agents = {prow: set() for _, prow, _ in data_rows}
        for s in matching_keys:
            for at in agent_types:
                if s is not None and at in s:
                    agents_data[at][s] = {}
                    for _, prow, _ in data_rows:
                        mean = round(dom_data[s][prow]["mean"], 1)
                        std = round(dom_data[s][prow]["std"], 1)
                        agents_data[at][s][prow] = (mean, std)
                    break
        for _, prow, cmp in data_rows:
            group_bests = {}
            overall_best_val = 999999999  # overall max
            group_best_val = 9999999999  # group max
            # find max values
            for at in agents_data:
                for agent in agents_data[at]:
                    val = agents_data[at][agent][prow][0]
                    if compare(val, overall_best_val, cmp):
                        overall_best_val = val
                    if compare(val, group_best_val, cmp):
                        group_best_val = val
                group_bests[at] = group_best_val

            # find max agents
            for at in agents_data:
                for agent in agents_data[at]:
                    val = agents_data[at][agent][prow][0]
                    if val == group_bests[at]:
                        group_best_agents[at][prow].add(agent)
                    if val == overall_best_val:
                        overall_best_agents[prow].add(agent)

        print(agents_data)
        print(group_best_agents)
        print(overall_best_agents)
        # print(ldom, pdom)
        # print(matching_keys)
        # row_data = {}
        for i, (lrow, prow, _) in enumerate(data_rows):
            #     for key in matching_keys:
            #         if key is None:
            #             row_data[key] = (-1, -1)
            #         else:
            #             row_data[key] = (
            #                 round(dom_data[key][prow]["mean"], 1),
            #                 round(dom_data[key][prow]["std"], 1),
            #             )
            #     # find uni/bidir maxes
            #     for typ in agent_types:
            #         max = -1
            #         for key in matching_keys:
            #             if key is None:
            #                 continue
            #             if typ in key:
            #                 if row_data[key][0] > max:
            #                     max = row_data[key][0]
            #
            coldata = []
            for key in matching_keys:
                if key is None:
                    coldata.append("")
                else:
                    agent_data = dom_data[key]
                    best_agents = None
                    for at, ba in group_best_agents.items():
                        if at in key:
                            best_agents = ba

                    print(key)
                    print(best_agents[prow])
                    if key in best_agents[prow]:
                        s = f"\\makecell{{\\textbf{{{agent_data[prow]['mean']:.1f}}}\\\\\\textbf{{({agent_data[prow]['std']:.1f})}}"
                    if key in overall_best_agents[prow]:
                        s = f"\\makecell{{\\textbf{{{agent_data[prow]['mean']:.1f}}}\\\\\\textbf{{({agent_data[prow]['std']:.1f})}}"
                    else:
                        s = f"\\makecell{{{agent_data[prow]['mean']:.1f}\\\\({agent_data[prow]['std']:.1f})}}"
                    coldata.append(s)

            if i == 0:
                first_col = f"\\multirow{{ 2}}{{*}}{{{ldom}}} & {lrow}"
            else:
                first_col = f"& {lrow}"
                # make col data

            data_cols = " & ".join(coldata)
            latex_table.append(f"{first_col} & {data_cols} \\\\")
        latex_table.append(r"\midrule")

    latex_table.extend(
        [
            r"% \bottomrule",
            r"\end{tabularx}",
            r"\caption{A test caption}",
            r"\label{table2}",
            r"\end{table}",
            r"\normalsize",
        ]
    )

    return "\n".join(latex_table)


if __name__ == "__main__":
    indir = Path("/home/ken/Projects/bilevin/tables")
    common_stats = {
        dom: pkl.load((indir / f"{dom}_common_stats.pkl").open("rb")) for _, dom in doms
    }
    s = create_common_table(common_stats, "common_stats")
    (indir / "common_table.tex").open("w").write(s)
