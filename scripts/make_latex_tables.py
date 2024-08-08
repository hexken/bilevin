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

data_rows = [("E", "exp"), ("L", "len")]
agent_columns = ("AStar", "BiAStar", "Levin", "BiLevin", "PHS", "BiPHS")


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
        matching_keys = [
            next((key for key in dom_data if key.startswith(prefix)), None)
            for prefix in agent_columns
        ]
        print(ldom, pdom)
        print(matching_keys)
        for i, (lrow, prow) in enumerate(data_rows):
            coldata = []
            for key in matching_keys:
                if key is None:
                    coldata.append("")
                else:
                    agent_data = dom_data[key]
                    coldata.append(
                        f"\\makecell{{{agent_data[prow]['mean']:.1f}\\\\({agent_data[prow]['std']:.1f})}}",
                    )

            if i == 0:
                first_col = f"\\multirow{{ 2}}{{*}}{{{ldom}}} & {lrow}"
            else:
                first_col = f"& {lrow}"
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
