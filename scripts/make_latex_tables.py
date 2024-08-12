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

common_data_rows = [("Expansions", "exp", "<"), ("Sol. Length", "len", "<")]
all_data_rows = [("Solved", "solved", ">"), ("Expansions", "total_exp", "<")]

agent_columns = ("AStar", "BiAStar", "Levin", "BiLevin", "PHS", "BiPHS")
agent_types = ("AStar", "Levin", "PHS")

agent_columns_bi = ("BiAStar", "BiLevin", "BiPHS")
bi_data_rows = [
    ("F-B Sol. Len.", "fb_len", "<"),
    ("F-B Avg. Action Prob.", "fb_ap_b", "<"),
    ("F-B Avg. H Error", "fb_he_b", "<"),
]


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


def create_bi_table(
    data: dict[str, pd.DataFrame], data_rows, n_decimals=3, show_solved=True
) -> str:
    round_fmt = f".{n_decimals}f"
    latex_table = [
        r"\begin{table}[htbp]",
        r"\footnotesize",
        r"\centering",
        r"\begin{tabular}{cc|ccc}",
        r"\phantom{} & \phantom{} & \mbiastar & \bilts & \biphs \\",
        r"\cmidrule{3-8}",
    ]

    for ldom, pdom in doms:
        overall_best_agents = {prow: set() for _, prow, _ in data_rows}
        group_best_agents = {}
        group_bests = {}
        dom_data = data[pdom]
        # get proper keys for each agent
        matching_keys = [
            next((key for key in dom_data if key.startswith(prefix)), None)
            for prefix in agent_columns_bi
        ]
        key_to_agent_type = {
            key: agent_type
            for key in matching_keys
            for agent_type in agent_types
            if (key is not None and agent_type in key)
        }

        # group proper keys by agent type
        agents_data = {at: {} for at in agent_types}
        for agent in matching_keys:
            if agent is not None:
                at = key_to_agent_type[agent]
                agents_data[at][agent] = {}
                for _, prow, _ in data_rows:
                    mean = round(dom_data[agent][prow]["mean"], n_decimals)
                    std = round(dom_data[agent][prow]["std"], n_decimals)
                    agents_data[at][agent][prow] = (mean, std)

        for _, prow, cmp in data_rows:
            if cmp == ">":
                overall_best_val = -1  # overall max
            else:
                overall_best_val = 99999999999  # overall max
            group_bests[prow] = {}
            group_best_agents[prow] = {at: set() for at in agent_types}
            # find max values
            for at in agents_data:
                if cmp == ">":
                    group_best_val = -1  # overall max
                else:
                    group_best_val = 99999999999  # overall max
                for agent in agents_data[at]:
                    val = agents_data[at][agent][prow][0]
                    if compare(val, overall_best_val, cmp):
                        overall_best_val = val
                    if compare(val, group_best_val, cmp):
                        group_best_val = val
                group_bests[prow][at] = group_best_val

            # find max agents
            for at in agents_data:
                for agent in agents_data[at]:
                    val = agents_data[at][agent][prow][0]
                    if val == group_bests[prow][at]:
                        group_best_agents[prow][at].add(agent)
                    if val == overall_best_val:
                        overall_best_agents[prow].add(agent)

        print(group_bests)
        print(agents_data)
        print(group_best_agents)
        print(overall_best_agents)
        # print(ldom, pdom)
        # print(matching_keys)
        # row_data = {}
        for i, (lrow, prow, _) in enumerate(data_rows):
            coldata = []
            for agent in matching_keys:
                if agent is None:
                    coldata.append("")
                else:
                    agent_data = dom_data[agent]
                    count = agent_data["exp"]["count"]
                    agent_type = key_to_agent_type[agent]
                    best_agents = group_best_agents[prow][agent_type]

                    # print(f"dom {pdom} row: {prow} key {agent} best: {best_agents}")
                    # if agent in overall_best_agents[prow]:
                    #     s = f"\\makecell{{\\underline{{\\textbf{{{agent_data[prow]['mean']:{round_fmt}}}}}}\\\\\\underline{{\\textbf{{({agent_data[prow]['std']:{round_fmt}})}}}}}}"
                    # elif agent in best_agents:
                    #     s = f"\\makecell{{\\textbf{{{agent_data[prow]['mean']:{round_fmt}}}}\\\\\\textbf{{({agent_data[prow]['std']:{round_fmt}})}}}}"
                    # else:
                    #     s = f"\\makecell{{{agent_data[prow]['mean']:{round_fmt}}\\\\({agent_data[prow]['std']:{round_fmt}})}}"
                    s = f"\\makecell{{{agent_data[prow]['mean']:{round_fmt}}\\\\({agent_data[prow]['std']:{round_fmt}})}}"
                    coldata.append(s)

            if i == 0:
                if show_solved:
                    first_col = f"\\multirow{{2}}{{*}}{{\parbox{{1.25cm}}{{{ldom} ({count})}}}} & {lrow}"
                else:
                    first_col = f"\\multirow{{2}}{{*}}{{{ldom}}} & {lrow}"
            else:
                first_col = f"& {lrow}"
                # make col data

            data_cols = " & ".join(coldata)
            latex_table.append(f"{first_col} & {data_cols} \\\\")
        latex_table.append(r"\midrule")

    latex_table.extend(
        [
            r"% \bottomrule",
            r"\end{tabular}",
            r"\caption{A test caption}",
            r"\label{table2}",
            r"\end{table}",
            r"\normalsize",
        ]
    )

    return "\n".join(latex_table)


def create_common_table(
    data: dict[str, pd.DataFrame], data_rows, n_decimals=1, show_solved=True
) -> str:
    round_fmt = f".{n_decimals}f"
    latex_table = [
        r"\begin{table}[htbp]",
        r"\footnotesize",
        r"\centering",
        r"\begin{tabular}{cc|cc|cc|cc}",
        r"\phantom{} & \phantom{} & \mastar & \mbiastar & \lts & \bilts & \phs & \biphs \\",
        r"\cmidrule{3-8}",
    ]

    for ldom, pdom in doms:
        overall_best_agents = {prow: set() for _, prow, _ in data_rows}
        group_best_agents = {}
        group_bests = {}
        dom_data = data[pdom]
        # get proper keys for each agent
        matching_keys = [
            next((key for key in dom_data if key.startswith(prefix)), None)
            for prefix in agent_columns
        ]
        key_to_agent_type = {
            key: agent_type
            for key in matching_keys
            for agent_type in agent_types
            if (key is not None and agent_type in key)
        }

        # group proper keys by agent type
        agents_data = {at: {} for at in agent_types}
        for agent in matching_keys:
            if agent is not None:
                at = key_to_agent_type[agent]
                agents_data[at][agent] = {}
                for _, prow, _ in data_rows:
                    mean = round(dom_data[agent][prow]["mean"], n_decimals)
                    std = round(dom_data[agent][prow]["std"], n_decimals)
                    agents_data[at][agent][prow] = (mean, std)

        for _, prow, cmp in data_rows:
            if cmp == ">":
                overall_best_val = -1  # overall max
            else:
                overall_best_val = 99999999999  # overall max
            group_bests[prow] = {}
            group_best_agents[prow] = {at: set() for at in agent_types}
            # find max values
            for at in agents_data:
                if cmp == ">":
                    group_best_val = -1  # overall max
                else:
                    group_best_val = 99999999999  # overall max
                for agent in agents_data[at]:
                    val = agents_data[at][agent][prow][0]
                    if compare(val, overall_best_val, cmp):
                        overall_best_val = val
                    if compare(val, group_best_val, cmp):
                        group_best_val = val
                group_bests[prow][at] = group_best_val

            # find max agents
            for at in agents_data:
                for agent in agents_data[at]:
                    val = agents_data[at][agent][prow][0]
                    if val == group_bests[prow][at]:
                        group_best_agents[prow][at].add(agent)
                    if val == overall_best_val:
                        overall_best_agents[prow].add(agent)

        print(group_bests)
        print(agents_data)
        print(group_best_agents)
        print(overall_best_agents)
        # print(ldom, pdom)
        # print(matching_keys)
        # row_data = {}
        for i, (lrow, prow, _) in enumerate(data_rows):
            coldata = []
            for agent in matching_keys:
                if agent is None:
                    coldata.append("")
                else:
                    agent_data = dom_data[agent]
                    count = agent_data["exp"]["count"]
                    agent_type = key_to_agent_type[agent]
                    best_agents = group_best_agents[prow][agent_type]

                    # print(f"dom {pdom} row: {prow} key {agent} best: {best_agents}")
                    if agent in overall_best_agents[prow]:
                        s = f"\\makecell{{\\underline{{\\textbf{{{agent_data[prow]['mean']:{round_fmt}}}}}}\\\\\\underline{{\\textbf{{({agent_data[prow]['std']:{round_fmt}})}}}}}}"
                    elif agent in best_agents:
                        s = f"\\makecell{{\\textbf{{{agent_data[prow]['mean']:{round_fmt}}}}\\\\\\textbf{{({agent_data[prow]['std']:{round_fmt}})}}}}"
                    else:
                        s = f"\\makecell{{{agent_data[prow]['mean']:{round_fmt}}\\\\({agent_data[prow]['std']:{round_fmt}})}}"
                    coldata.append(s)

            if i == 0:
                if show_solved:
                    first_col = f"\\multirow{{2}}{{*}}{{\parbox{{1.25cm}}{{{ldom} ({count})}}}} & {lrow}"
                else:
                    first_col = f"\\multirow{{2}}{{*}}{{{ldom}}} & {lrow}"
            else:
                first_col = f"& {lrow}"
                # make col data

            data_cols = " & ".join(coldata)
            latex_table.append(f"{first_col} & {data_cols} \\\\")
        latex_table.append(r"\midrule")

    latex_table.extend(
        [
            r"% \bottomrule",
            r"\end{tabular}",
            r"\caption{A test caption}",
            r"\label{table2}",
            r"\end{table}",
            r"\normalsize",
        ]
    )

    return "\n".join(latex_table)


if __name__ == "__main__":
    indir = Path("/home/ken/Envs/tables")
    common_stats = {
        dom: pkl.load((indir / f"{dom}_common_stats.pkl").open("rb")) for _, dom in doms
    }
    s = create_common_table(common_stats, common_data_rows, n_decimals=1)
    (indir / "common_table.tex").open("w").write(s)

    solved_stats = {
        dom: pkl.load((indir / f"{dom}_solved_stats.pkl").open("rb")) for _, dom in doms
    }
    s = create_common_table(
        solved_stats, all_data_rows, n_decimals=0, show_solved=False
    )
    (indir / "all_table.tex").open("w").write(s)

    s = create_bi_table(solved_stats, bi_data_rows, n_decimals=3, show_solved=True)
    (indir / "bi_table.tex").open("w").write(s)
