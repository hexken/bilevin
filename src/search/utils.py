from __future__ import annotations
import math
import random
from typing import Optional, TYPE_CHECKING
import warnings

import numpy as np
import pandas as pd
import torch as to
from torch import Tensor


if TYPE_CHECKING:
    from domains.domain import Domain, State


class Problem:
    def __init__(self, id: int, domain: Domain):
        self.id: int = id
        self.domain: Domain = domain

    def __hash__(self):
        return self.id.__hash__()

    def __eq__(self, other):
        return self.id == other.id


search_result_header = [
    "id",
    "time",
    "exp",
    "fexp",
    "bexp",
    "len",
    "fg",
    "bg",
    "fpnll",
    "bpnll",
]

int_columns = ["exp", "fexp", "bexp"]


def print_model_train_summary(
    model_train_df,
    bidirectioal: bool,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        floss = model_train_df["floss"].mean()
        facc = model_train_df["facc"].mean()
        print(f"\nFloss: {floss:.3f}")
        print(f"Facc: {facc:.3f}")
        if bidirectioal:
            bloss = model_train_df["bloss"].mean()
            bacc = model_train_df["bacc"].mean()
            print(f"Bloss: {bloss:.3f}")
            print(f"Bacc: {bacc:.3f}")


def print_search_summary(
    search_df,
    bidirectional: bool,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        solved_df = search_df.dropna()
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
        if bidirectional:
            fb_exp = (solved_df["fexp"] / solved_df["bexp"]).mean()
            fb_lens = (solved_df["fg"] / solved_df["bg"]).mean()
            fb_pnll = (solved_df["fpnll"] / solved_df["bpnll"]).mean()
            print(f"\nFB Exp: {fb_exp:.3f}")
            print(f"FB Len: {fb_lens:.3f}")
            print(f"FB Pnll: {fb_pnll:.3f}")


class SearchNode:
    def __init__(
        self,
        state: State,
        parent: Optional[SearchNode],
        parent_action: Optional[int],
        actions: list[int],
        actions_mask: Tensor,
        g: int,
        log_prob: float,
        log_action_probs: Optional[Tensor] = None,
        action_hs: Optional[Tensor] = None,
        h: Optional[float] = None,
        f: Optional[float] = None,
    ):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.g = g
        self.log_prob = log_prob
        self.actions = actions
        self.actions_mask = actions_mask
        self.log_action_probs = log_action_probs
        self.action_hs = action_hs
        self.h = h
        self.f = g if f is None else f

    def __eq__(self, other):
        """
        Verify if two SearchNodes are identical by verifying the
         state in the nodes.
        """
        return self.state == other.state

    def __lt__(self, other):
        """
        less-than used by the heap
        """
        return self.f < other.f

    def __hash__(self):
        """
        Hash function used in the closed list
        """
        return self.state.__hash__()


class Trajectory:
    def __init__(
        self,
        states: Tensor,
        actions: Tensor,
        masks: Tensor,
        num_expanded: int,
        partial_g_cost: int,  # g_cost of node that generated sol.
        partial_log_prob: float,  # probability of node that generates sol.
        goal_state_t: Optional[Tensor] = None,
        forward: bool = True,
    ):
        self.states = states
        self.actions = actions
        self.num_expanded = num_expanded
        self.partial_g_cost = partial_g_cost
        self.partial_log_prob = partial_log_prob
        self.masks = masks
        self.goal_state_t = goal_state_t
        self.forward = forward

        self._len = len(self.actions)

    @classmethod
    def from_goal_node(
        cls,
        domain: Domain,
        goal_node: SearchNode,
        num_expanded: int,
        partial_g_cost: Optional[int] = None,
        partial_log_prob: Optional[float] = None,
        goal_state_t: Optional[Tensor] = None,
        forward: bool = True,
    ) -> Trajectory:
        """
        Receives a SearchNode representing a solution to the problem.
        Backtracks the path performed by search, collecting state-action pairs along the way.
        """
        assert domain.is_goal(goal_node.state)
        goal_state_t = goal_state_t.unsqueeze(0) if goal_state_t is not None else None
        action = goal_node.parent_action
        node = goal_node.parent

        if partial_g_cost is None:
            partial_g_cost = goal_node.g
        if partial_log_prob is None:
            partial_log_prob = goal_node.log_prob

        states = []
        actions = []
        masks = []

        while node:
            state_t = domain.state_tensor(node.state)
            states.append(state_t)
            actions.append(action)
            masks.append(node.actions_mask)
            action = node.parent_action
            node = node.parent

        states = to.stack(tuple(reversed(states)))
        actions = to.tensor(tuple(reversed(actions)))
        masks = to.stack(tuple(reversed(masks)))

        return cls(
            states=states,
            actions=actions,
            masks=masks,
            num_expanded=num_expanded,
            partial_g_cost=partial_g_cost,
            partial_log_prob=partial_log_prob,
            goal_state_t=goal_state_t,
            forward=forward,
        )

    def __len__(self):
        return self._len


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    to.manual_seed(seed)
