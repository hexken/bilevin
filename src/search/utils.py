from __future__ import annotations
import math
from typing import Optional, TYPE_CHECKING
import warnings

import numpy as np
import torch as to
from torch import Tensor
from torch.nn.functional import nll_loss

from models.models import AgentModel

if TYPE_CHECKING:
    from domains.domain import Domain, State


class Problem:
    def __init__(self, id: str, domain: Domain):
        self.id = id
        self.domain = domain

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
    "fnll",
    "bnll",
]

train_csvfields = [
    "epoch",
    "floss",
    "facc",
    "bloss",
    "bacc",
    "solved",
    "sol_len",
    "cum_uniq_solved",
    "exp_ratio",
    "fb_exp_ratio",
    "fb_g_ratio",
    "fb_pnll_ratio",
    "fb_nll_ratio",
]

test_csvfields = [
    "epoch",
    "solved",
    "sol_len",
    "exp_ratio",
    "fb_exp_ratio",
    "fb_g_ratio",
    "fb_pnll_ratio",
    "fb_nll_ratio",
]

int_columns = ["Exp", "FExp", "BExp"]


def print_model_train_summary(
    model_train_df,
    bidirectioal: bool,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        floss = model_train_df["floss"].mean()
        facc = model_train_df["facc"].mean()
        print(f"Floss: {floss:.3f}")
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
        solved_df = search_df[(search_df["flen"] > 0) | (search_df["blen"] > 0)]
        solved = len(solved_df) / len(search_df)
        time = solved_df["time"].mean()
        exp = (solved_df["fexp"] + solved_df["bexp"]).mean()
        if bidirectional:
            fb_exp = (solved_df["fexp"] / solved_df["bexp"]).mean()
            fb_lens = (solved_df["flen"] / solved_df["blen"]).mean()
            fb_pnll = (solved_df["fpnll"] / solved_df["bpnll"]).mean()
            fb_nll = (solved_df["fnll"] / solved_df["bnll"]).mean()
            lens = (solved_df["flen"] + solved_df["blen"]).mean()
        else:
            lens = solved_df["flen"].mean()
        print(f"Solved: {solved:.3f}")
        print(f"Time: {time:.3f}")
        print(f"Exp: {exp:.3f}")
        print(f"Len: {lens:.3f}")
        if bidirectional:
            print(f"FB Exp: {fb_exp:.3f}")
            print(f"FB Len: {fb_lens:.3f}")
            print(f"FB Pnll: {fb_pnll:.3f}")
            print(f"FB Nll: {fb_nll:.3f}")


class SearchNode:
    def __init__(
        self,
        state: State,
        g_cost: int,
        parent: Optional[SearchNode] = None,
        parent_action: Optional[int] = None,
        log_prob: Optional[float] = None,
        actions: Optional[list[int]] = None,
        actions_mask: Optional[Tensor] = None,
        log_action_probs: Optional[Tensor] = None,
        cost: Optional[float] = None,
    ):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.g_cost = g_cost
        self.log_prob = log_prob
        self.actions = actions
        self.actions_mask = actions_mask
        self.log_action_probs = log_action_probs
        self.cost = g_cost if cost is None else cost

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
        return self.cost < other.cost

    def __hash__(self):
        """
        Hash function used in the closed list
        """
        return self.state.__hash__()


def levin_cost(node: SearchNode):
    return math.log(node.g_cost) - node.log_prob  # type:ignore


class Trajectory:
    def __init__(
        self,
        states: Tensor,
        actions: Tensor,
        masks: Tensor,
        num_expanded: int,
        partial_g_cost: Optional[int] = None,  # g_cost of node that generated sol.
        partial_log_prob: Optional[
            float
        ] = None,  # probability of node that generates sol.
        log_prob: Optional[float] = None,
        goal_state_t: Optional[Tensor] = None,
        forward: bool = True,
    ):
        self.states = states
        self.actions = actions
        self.num_expanded = num_expanded
        self.partial_g_cost = partial_g_cost
        self.partial_log_prob = partial_log_prob
        self.log_prob = log_prob
        self.masks = masks
        self.goal_state_t = goal_state_t
        self.forward = forward

        self.cost = len(self.actions)
        self.costs_to_go = to.arange(self.cost, 0, -1)
        self._len = len(self.actions)

    @classmethod
    def from_goal_node(
        cls,
        domain: Domain,
        final_node: SearchNode,
        num_expanded: int,
        partial_g_cost: int,
        partial_log_prob: Optional[float] = None,
        log_prob: Optional[float] = None,
        model: Optional[AgentModel] = None,
        goal_state_t: Optional[Tensor] = None,
        forward: bool = True,
    ):
        """
        Receives a SearchNode representing a solution to the problem.
        Backtracks the path performed by search, collecting state-action pairs along the way.
        """
        goal_state_t = goal_state_t.unsqueeze(0) if goal_state_t is not None else None
        action = final_node.parent_action
        node = final_node.parent

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

        if model:
            with to.no_grad():
                # k = partial_g_cost
                log_probs, _ = model(
                    states,
                    mask=masks,
                    forward=forward,
                    goal_state_t=goal_state_t,
                )
                # partial_nll = nll_loss(
                #     log_probs[:k], actions[:k], reduction="sum"
                # ).item()
                # nll = (
                #     partial_nll
                #     + nll_loss(log_probs[k:], actions[k:], reduction="sum").item()
                # )

            lp = -nll_loss(log_probs, actions, reduction="sum").item()
            # plp = -partial_nll

            if log_prob and not np.isclose(log_prob, lp):
                print(f"Warning: search log_prob != model log_prob {log_prob} {lp}")
            else:
                log_prob = lp
            # if partial_log_prob and not np.isclose(partial_log_prob, plp):
            #     print(
            #         f"Warning: search partial_log_prob != model partial_log_prob {partial_log_prob} {plp}"
            #     )
            # else:
            #     partial_log_prob = plp

        return cls(
            states=states,
            actions=actions,
            masks=masks,
            num_expanded=num_expanded,
            partial_g_cost=partial_g_cost,
            partial_log_prob=partial_log_prob,
            log_prob=log_prob,
            goal_state_t=goal_state_t,
            forward=forward,
        )

    def __len__(self):
        return self._len

    def get_subgoal_trajs(self: Trajectory):
        """
        Generates all sub-trajectories of a trajectory.
        """
        sub_trajs = []
        for goal_idx in range(len(self) - 1, 0, -1):
            masks = self.masks[:goal_idx]

            est_num_exp = int(self.num_expanded / (len(self) - goal_idx + 1))
            sub_trajs.append(
                Trajectory(
                    self.states[:goal_idx],
                    self.actions[:goal_idx],
                    masks=masks,
                    num_expanded=est_num_exp,
                    forward=False,
                    goal_state_t=self.states[goal_idx].unsqueeze(0),
                )
            )
        return sub_trajs
