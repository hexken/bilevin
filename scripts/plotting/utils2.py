from collections import OrderedDict
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import pickle as pkl
import re
import subprocess

import matplotlib as mpl
from natsort import natsorted
import pandas as pd


def add_sum_fb_exp(df: pd.DataFrame) -> pd.DataFrame:
    if "fexp" in df.columns and "bexp" in df.columns:
        df["exp"] = df["fexp"] + df["bexp"]
    else:
        df["exp"] = df["fexp"]
    return df


def int_cols_to_float(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == pd.UInt32Dtype() or df[col].dtype == pd.UInt64Dtype():
            df[col] = df[col].astype(float)
    return df


class LineStyleMapper:
    def __init__(self):
        self.uni_marker = "o"
        self.bi_marker = "x"
        self.uni_ls = "-"
        self.bi_lds = "--"
        self.bibfs_ls = "--"
        self.bialt_ls = ":"
        self.bi_hatch = "||"
        self.uni_hatch = None

    def get_ls(self, s: str):
        if "AStar" in s:
            c = "#b1201b"
        elif "Levin" in s:
            # e43b35 an orange
            c = "#25b44e"
        elif "PHS" in s:
            # lighter blue #2941df
            c = "#1823b2"
        else:
            raise ValueError(f"Unknown algorithm: {s}")

        # if "Alt" in s:
        #     ls = self.bialt_ls
        # elif "BFS" in s:
        #     ls = self.bibfs_ls
        # else:
        #     ls = self.uni_ls

        if "Bi" in s:
            ls = self.bibfs_ls
            h = self.bi_hatch
        else:
            ls = self.uni_ls
            h = self.uni_hatch

        return c, ls, h
