from __future__ import annotations
import warnings

import numpy as np
import pandas as pd

from search.agent import Agent


int_columns = ["id", "len", "fg", "bg", "fexp", "bexp"]

search_result_header = (
    "id",
    "time",
    "fexp",
    "bexp",
    "len",
    "fg",
    "fap",
    "facc",
    "fhe",
    "bg",
    "bap",
    "bacc",
    "bhe",
)


class ResultsLog:
    def __init__(
        self,
        agent: Agent | None = None,
        policy_based: bool | None = None,
        heuristic_based: bool | None = None,
        bidirectional: bool | None = None,
    ):
        self.data = {key: [] for key in search_result_header}
        self.solved = 0

        if agent is not None:
            self.policy_based = agent.has_policy
            self.heuristic_based = agent.has_heuristic
            self.bidirectional = agent.is_bidirectional
        else:
            self.policy_based = policy_based
            self.heuristic_based = heuristic_based
            self.bidirectional = bidirectional
        self.solved = 0

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
        ret_df = mod_df(
            ret_df, self.policy_based, self.heuristic_based, self.bidirectional, sort
        )
        return ret_df

    def __getitem__(self, key) -> ResultsLog:
        ret = ResultsLog(
            policy_based=self.policy_based,
            heuristic_based=self.heuristic_based,
            bidirectional=self.bidirectional,
        )
        data = {k: v[key] for k, v in self.data.items()}
        solved = len(data["len"]) - sum(l for l in data["len"] if l > 0)
        ret.load_state_dict({"data": data, "solved": solved})
        return ret

    def __len__(self):
        return len(self.data["id"])


class Result:
    def __init__(self, id, time, fexp, bexp, len, f_traj=None, b_traj=None):
        self.id = id
        self.time = time
        self.fexp = fexp
        self.bexp = bexp
        self.len = len
        self.f_traj = f_traj
        self.b_traj = b_traj

        if self.f_traj is not None:
            self.fg = self.f_traj.partial_g_cost
            self.fap = self.f_traj.avg_action_prob
            self.facc = self.f_traj.acc
            self.fhe = self.f_traj.avg_h_abs_error
        else:
            self.fg = np.nan
            self.fap = np.nan
            self.facc = np.nan
            self.fhe = np.nan

        if self.b_traj is not None:
            self.bg = self.b_traj.partial_g_cost
            self.bap = self.b_traj.avg_action_prob
            self.bacc = self.b_traj.acc
            self.bhe = self.b_traj.avg_h_abs_error
        else:
            self.bg = np.nan
            self.bap = np.nan
            self.bacc = np.nan
            self.bhe = np.nan


def mod_df(ret_df, policy_based, heuristic_based, bidirectional, sort=False):
    if bidirectional:
        exp = ret_df["fexp"] + ret_df["bexp"]
    else:
        exp = ret_df["fexp"]
    for col in int_columns:
        ret_df[col] = ret_df[col].astype(pd.UInt32Dtype())
    ret_df.insert(2, "exp", exp)
    if not policy_based:
        ret_df = ret_df.drop(columns=["facc", "fap", "bap", "bacc"], errors="ignore")
    if not heuristic_based:
        ret_df = ret_df.drop(columns=["fhe", "bhe"], errors="ignore")
    if not bidirectional:
        ret_df = ret_df.drop(
            columns=["bexp", "bg", "bacc", "bap", "bhe"], errors="ignore"
        )
    if sort:
        ret_df = ret_df.sort_values("exp")
    return ret_df


def print_model_train_summary(
    model_train_df,
    bidirectioal: bool,
    policy_based: bool,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        floss = model_train_df["floss"].mean()
        print(f"\nFloss: {floss:.3f}")
        if bidirectioal:
            bloss = model_train_df["bloss"].mean()
            print(f"Bloss: {bloss:.3f}")


def print_search_summary(
    search_df,
    bidirectional: bool,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        solved_df = search_df.dropna(subset=["len"])
        solved = len(solved_df) / len(search_df)
        time = solved_df["time"].mean()
        if bidirectional:
            s_exp = (solved_df["fexp"] + solved_df["bexp"]).mean()
        else:
            s_exp = (solved_df["fexp"]).mean()
        lens = solved_df["len"].mean()
        print(f"Solved: {len(solved_df)}/{len(search_df)} ({solved * 100:.2f}%)")
        print(f"Time: {time:.2f} ({search_df['time'].mean():.2f})")
        total_exp = search_df["exp"].sum()
        print(f"Exp: {s_exp:.2f} ({total_exp/len(search_df):.2f})")
        print(f"Len: {lens:.2f}")
        if "fap" in solved_df.columns:
            fap = solved_df["fap"].mean()
            print(f"Fap: {fap:.3f}")
        if "facc" in solved_df.columns:
            facc = solved_df["facc"].mean()
            print(f"Facc: {facc:.3f}")
        if "fhe" in solved_df.columns:
            fhe = solved_df["fhe"].mean()
            print(f"Fhe: {fhe:.3f}")
        if "bap" in solved_df.columns:
            bap = solved_df["bap"].mean()
            print(f"Bap: {bap:.3f}")
        if "bacc" in solved_df.columns:
            bacc = solved_df["bacc"].mean()
            print(f"Bacc: {bacc:.3f}")
        if "bhe" in solved_df.columns:
            bhe = solved_df["bhe"].mean()
            print(f"Bhe: {bhe:.3f}")
        if bidirectional:
            s_fb_exp = solved_df["fexp"] / solved_df["bexp"]
            s_fb_exp = s_fb_exp[s_fb_exp != np.inf].mean()

            a_fb_exp = search_df["fexp"] / search_df["bexp"]
            a_fb_exp = a_fb_exp[a_fb_exp != np.inf].mean()

            fb_lens = solved_df["fg"] / (solved_df["bg"] + solved_df["fg"])
            fb_lens = fb_lens[fb_lens != np.inf].mean()

            if "facc" in solved_df.columns:
                fb_acc = solved_df["facc"] / solved_df["bacc"]
                fb_acc = fb_acc[fb_acc != np.inf].mean()
                print(f"\nFB Acc: {s_fb_exp:.3f}")
            else:
                print()
            print(f"FB Exp: {s_fb_exp:.3f} ({a_fb_exp:.3f})")
            print(f"FB Len: {fb_lens:.3f}")
    return len(solved_df), total_exp
