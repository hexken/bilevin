from __future__ import annotations
import warnings

import numpy as np
import pandas as pd

int_columns = ["id", "len", "exp", "fg", "bg"]

search_result_header = (
    "id",
    "time",
    "exp",
    "len",
)


class ResultsLog:
    def __init__(self, results: list[Result] | None = None):
        self.data = {key: [] for key in search_result_header}
        self.solved = 0
        if results is not None:
            self.append(results)

    def state_dict(self):
        return {"data": self.data, "solved": self.solved}

    def load_state_dict(self, state: dict):
        self.data = state["data"]
        self.solved = state["solved"]

    def clear(self):
        for k in self.data.keys():
            self.data[k].clear()
        self.solved = 0

    def append(self, result: Result | list[Result]):
        if isinstance(result, list):
            for res in result:
                self._append(res)
        else:
            self._append(result)

    def _append(self, result: Result):
        for key in search_result_header:
            self.data[key].append(result.__dict__[key])
        if result.len > 0:
            self.solved += 1

    def get_df(self, sort=False):
        ret_df = pd.DataFrame(self.data)
        if sort:
            ret_df = ret_df.sort_values("exp")
        for col in int_columns:
            if col in ret_df.columns:
                ret_df[col] = ret_df[col].astype(pd.UInt32Dtype())
        return ret_df

    def __getitem__(self, key) -> ResultsLog:
        ret = ResultsLog()
        data = {k: v[key] for k, v in self.data.items()}
        solved = len(data["len"]) - sum(l for l in data["len"] if l > 0)
        ret.load_state_dict({"data": data, "solved": solved})
        return ret

    def __len__(self):
        return len(self.data["id"])


class Result:
    def __init__(self, id, time, exp, len, f_traj=None, b_traj=None):
        self.id = id
        self.time = time
        self.exp = exp
        self.len = len
        self.f_traj = f_traj
        self.b_traj = b_traj


def print_search_summary(
    search_df,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        solved_df = search_df.dropna(subset=["len"])
        solved = len(solved_df) / len(search_df)
        time = solved_df["time"].mean()
        # if bidirectional:
        #     s_exp = (solved_df["fexp"] + solved_df["bexp"]).mean()
        # else:
        #     s_exp = (solved_df["fexp"]).mean()
        lens = solved_df["len"].mean()
        exp = solved_df["exp"].mean()
        print(f"Solved: {len(solved_df)}/{len(search_df)} ({solved * 100:.2f}%)")
        print(f"Time: {time:.2f} ({search_df['time'].mean():.2f})")
        total_exp = search_df["exp"].sum()
        print(f"Exp: {exp:.2f} ({total_exp/len(search_df):.2f})")
        print(f"Len: {lens:.2f}")
    return len(solved_df), total_exp
