from __future__ import annotations
import random
import warnings

import numpy as np
import pandas as pd
import torch as to

from search.agent import Agent


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
        results: dict | None,
        agent: Agent | None = None,
        policy_based: bool | None = None,
        heuristic_based: bool | None = None,
        bidirectional: bool | None = None,
    ):
        if results is not None:
            self.results = results
        else:
            self.results = {key: [] for key in search_result_header}

        if agent is not None:
            self.policy_based = agent.has_policy
            self.heuristic_based = agent.has_heuristic
            self.bidirectional = agent.is_bidirectional
        else:
            self.policy_based = policy_based
            self.heuristic_based = heuristic_based
            self.bidirectional = bidirectional

    def append(self, result: Result | list[Result]):
        if isinstance(result, list):
            for res in result:
                self._append(res)
        else:
            self._append(result)

    def _append(self, result: Result):
        for key in search_result_header:
            self.results[key].append(result.__dict__[key])

    def get_df(self):
        ret_df = pd.DataFrame(self.results)
        ret_df = mod_df(
            ret_df, self.policy_based, self.heuristic_based, self.bidirectional
        )
        return ret_df

    def __getitem__(self, key) -> ResultsLog:
        return ResultsLog(
            {k: v[key] for k, v in self.results.items()},
            None,
            self.policy_based,
            self.heuristic_based,
            self.bidirectional,
        )

    def __len__(self):
        return len(self.results["id"])


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

    # @classmethod
    # def df_attrs(cls):
    #     return search_result_header

    # def __iter__(self):
    #     data = tuple(self.__dict__[var] for var in SearchResult.df_attrs())
    #     return iter(data)

    # @classmethod
    # def list_to_df(
    #     cls, results: list[SearchResult], policy_based, heuristic_based, bidirectional
    # ):
    #     ret_df = pd.DataFrame([item for item in results])
    #     ret_df.columns = cls.df_attrs()

    #     return ret_df


int_columns = ["id", "len", "fg", "bg", "fexp", "bexp"]


def mod_df(ret_df, policy_based, heuristic_based, bidirectional):
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
            exp = (solved_df["fexp"] + solved_df["bexp"]).mean()
        else:
            exp = (solved_df["fexp"]).mean()
        lens = solved_df["len"].mean()
        print(f"Problems: {len(search_df)}")
        print(f"Solved: {solved:.3f}")
        print(f"Time: {time:.3f}")
        print(f"Exp: {exp:.3f}")
        print(f"Len: {lens:.3f}")
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
            fb_exp = solved_df["fexp"] / solved_df["bexp"]
            fb_exp = fb_exp[fb_exp != np.inf].mean()
            fb_lens = solved_df["fg"] / solved_df["bg"]
            fb_lens = fb_lens[fb_lens != np.inf].mean()
            if "facc" in solved_df.columns:
                fb_acc = solved_df["facc"] / solved_df["bacc"]
                fb_acc = fb_acc[fb_acc != np.inf].mean()
                print(f"\nFB Acc: {fb_exp:.3f}")
            else:
                print()
            print(f"FB Exp: {fb_exp:.3f}")
            print(f"FB Len: {fb_lens:.3f}")


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    to.manual_seed(seed)
