from __future__ import annotations
from math import ceil
from typing import TYPE_CHECKING

import numpy as np
import torch as to
from torch import clamp, exp, log, sum
import torch.nn as nn
from torch.nn.functional import cross_entropy, log_softmax, nll_loss, normalize
from torch.nn.functional import mse_loss as mse

from search.traj import MetricTrajectory, Trajectory

if TYPE_CHECKING:
    from models.models import PolicyOrHeuristicModel, BYOL


def _byol_loss(x, y):
    x = normalize(x, dim=-1, p=2)
    y = normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def byol_loss(f_traj: Trajectory, b_traj: Trajectory, model: BYOL):
    pass


def metric_loss(
    f_traj: MetricTrajectory,
    b_traj: MetricTrajectory,
    model: PolicyOrHeuristicModel,
    adj_consistency: bool = False,
    ends_consistency: bool = False,
    n_samples: int = 0,
    children_weight=1.0,
    adj_weight=1.0,
    ends_weight=1.0,
    samples_weight=1.0,
):
    # todo wip, see if normalizing works
    loss = 0.0
    n = len(f_traj)
    # compute all necessary features
    f_states_feats = model(f_traj.states)
    b_states_feats = model(b_traj.states, forward=False)

    f_children_feats = [model(children) for children in f_traj.children]
    b_children_feats = [model(children, forward=False) for children in b_traj.children]

    # make a node and it's childrens features similar while keeping their distances 1
    # forward
    forw_c_loss = 0.0
    for feats, children_feats in zip(f_states_feats, f_children_feats):
        sq_dist = (children_feats - feats).pow(2).sum()
        forw_c_loss += sq_dist
        forw_c_loss += children_weight * (sq_dist - 1.0) ** 2

    forw_c_loss /= n

    # backward
    backw_c_loss = 0.0
    for feats, children_feats in zip(b_states_feats, b_children_feats):
        sq_dist = (children_feats - feats).pow(2).sum()
        backw_c_loss += sq_dist
        backw_c_loss += children_weight * (sq_dist - 1.0) ** 2
    backw_c_loss /= n

    # make the forward and backward model features of consecutive states along the solution
    # trajectory similar, while keeping their distances 1
    # redundent when the forward and backward models share all parameters
    if adj_consistency:
        for i in range(len(f_traj.states) - 1):
            sq_dist = (f_states_feats[i] - b_states_feats[n - i - 2]).pow(2).sum()
            loss += sq_dist
            loss += adj_weight * (sq_dist - 1.0) ** 2

    # make sq dist between forward feats of start node and backward feats of end node
    # <= sq number of actions in the traj
    ends_loss = 0.0
    if ends_consistency:
        sq_traj_dist = (len(f_traj.states) - 1) ** 2
        sq_dist = (f_states_feats[-1] - b_states_feats[0]).pow(2).sum()
        # loss += -sq_dist
        ends_loss += ends_weight * clamp(sq_traj_dist - sq_dist, min=0) ** 2
    ends_loss /= 2.0

    # sample pairs of states in the solution traj and add constraint that their features
    # sq dist <= sq dist in the traj
    sample_loss = 0.0
    if n_samples > 0 and n > 2:
        # make sure samples at least two actions separated
        if n > 2:
            n_samples = n - 2
        indices = np.arange(n - 1)
        s1_indices = np.random.choice(indices, n_samples, replace=False)
        s2_indices = np.random.randint(s1_indices, n, n_samples)
        for i, j in zip(s1_indices, s2_indices):
            sq_traj_dist = (j - i) ** 2
            sq_dist = (f_states_feats[i] - b_states_feats[n - j - 1]).pow(2).sum()
            # loss += -sq_dist
            sample_loss += samples_weight * clamp(sq_traj_dist - sq_dist, min=0) ** 2
        sample_loss /= n_samples

    loss = forw_c_loss + backw_c_loss + loss + ends_loss + sample_loss

    return loss


def mse_loss(traj: Trajectory, model: PolicyOrHeuristicModel, weight=1.0):
    _, h = model(
        traj.states,
        forward=traj.forward,
        goal_state_t=traj.goal_state_t,
        mask=traj.masks,
    )
    loss = mse(h, traj.cost_to_gos.unsqueeze(1))

    return loss, 0.0, 0.0


def cross_entropy_loss(traj: Trajectory, model: PolicyOrHeuristicModel):
    n_actions = len(traj)
    log_probs, _ = model(
        traj.states,
        forward=traj.forward,
        goal_state_t=traj.goal_state_t,
        mask=traj.masks,
    )
    loss = nll_loss(log_probs, traj.actions, reduction="mean")

    avg_action_nll = loss.item()
    acc = (log_probs.detach().argmax(dim=1) == traj.actions).sum().item() / n_actions

    return loss, avg_action_nll, acc


def cross_entropy_mse_loss(traj: Trajectory, model: PolicyOrHeuristicModel, weight=1.0):
    n_actions = len(traj)
    log_probs, hs = model(
        traj.states,
        forward=traj.forward,
        goal_state_t=traj.goal_state_t,
        mask=traj.masks,
    )
    ce_loss = nll_loss(log_probs, traj.actions, reduction="mean")

    avg_action_nll = ce_loss.item()
    acc = (log_probs.detach().argmax(dim=1) == traj.actions).sum().item() / n_actions

    mse_loss = mse(hs, traj.cost_to_gos.unsqueeze(1))
    loss = ce_loss + weight * mse_loss

    return loss, avg_action_nll, acc


def levin_avg_mse_loss(traj: Trajectory, model: PolicyOrHeuristicModel, weight=1.0):
    n_actions = len(traj)
    log_probs, hs = model(
        traj.states,
        forward=traj.forward,
        goal_state_t=traj.goal_state_t,
        mask=traj.masks,
    )
    loss = nll_loss(log_probs, traj.actions, reduction="mean")

    avg_action_nll = loss.item()
    acc = (log_probs.detach().argmax(dim=1) == traj.actions).sum().item() / n_actions

    mse_loss = mse(hs, traj.cost_to_gos.unsqueeze(1))
    loss = loss * traj.num_expanded + weight * mse_loss

    return loss, avg_action_nll, acc


def traj_nll_mse_loss(traj: Trajectory, model: PolicyOrHeuristicModel, weight=1.0):
    n_actions = len(traj)
    log_probs, hs = model(
        traj.states,
        forward=traj.forward,
        goal_state_t=traj.goal_state_t,
        mask=traj.masks,
    )
    loss = nll_loss(log_probs, traj.actions, reduction="sum")

    avg_action_nll = loss.item() / n_actions
    acc = (log_probs.detach().argmax(dim=1) == traj.actions).sum().item() / n_actions

    mse_loss = mse(hs, traj.cost_to_gos.unsqueeze(1))
    loss = loss + weight * mse_loss

    return loss, avg_action_nll, acc


def levin_sum_mse_loss(traj: Trajectory, model: PolicyOrHeuristicModel, weight=1.0):
    n_actions = len(traj)
    log_probs, hs = model(
        traj.states,
        forward=traj.forward,
        goal_state_t=traj.goal_state_t,
        mask=traj.masks,
    )
    loss = nll_loss(log_probs, traj.actions, reduction="sum")

    avg_action_nll = loss.item() / n_actions
    acc = (log_probs.detach().argmax(dim=1) == traj.actions).sum().item() / n_actions

    mse_loss = mse(hs, traj.cost_to_gos.unsqueeze(1))
    loss = loss * traj.num_expanded + weight * mse_loss

    return loss, avg_action_nll, acc


def levin_avg_loss(traj: Trajectory, model: PolicyOrHeuristicModel):
    n_actions = len(traj)
    log_probs, hs = model(
        traj.states,
        forward=traj.forward,
        goal_state_t=traj.goal_state_t,
        mask=traj.masks,
    )
    loss = nll_loss(log_probs, traj.actions, reduction="mean")

    avg_action_nll = loss.item()
    acc = (log_probs.detach().argmax(dim=1) == traj.actions).sum().item() / n_actions

    loss = loss * traj.num_expanded

    return loss, avg_action_nll, acc


def traj_nll_loss(traj: Trajectory, model: PolicyOrHeuristicModel):
    n_actions = len(traj)
    log_probs, hs = model(
        traj.states,
        forward=traj.forward,
        goal_state_t=traj.goal_state_t,
        mask=traj.masks,
    )
    loss = nll_loss(log_probs, traj.actions, reduction="sum")

    avg_action_nll = loss.item() / n_actions
    acc = (log_probs.detach().argmax(dim=1) == traj.actions).sum().item() / n_actions

    return loss, avg_action_nll, acc


def levin_sum_loss(traj: Trajectory, model: PolicyOrHeuristicModel):
    n_actions = len(traj)
    log_probs, hs = model(
        traj.states,
        forward=traj.forward,
        goal_state_t=traj.goal_state_t,
        mask=traj.masks,
    )
    loss = nll_loss(log_probs, traj.actions, reduction="sum")

    avg_action_nll = loss.item() / n_actions
    acc = (log_probs.detach().argmax(dim=1) == traj.actions).sum().item() / n_actions

    loss = loss * traj.num_expanded

    return loss, avg_action_nll, acc


def cross_entropy_mid_loss(traj: Trajectory, model: PolicyOrHeuristicModel):
    mid_idx = ceil(len(traj) / 2)
    mask = traj.masks[:mid_idx]
    actions = traj.actions[:mid_idx]
    states = traj.states[:mid_idx]
    log_probs, _ = model(
        states,
        forward=traj.forward,
        goal_state_t=traj.goal_state_t,
        mask=mask,
    )
    loss = nll_loss(log_probs, actions, reduction="mean")

    avg_action_nll = loss.item()
    acc = (log_probs.detach().argmax(dim=1) == actions).sum().item() / mid_idx

    return loss, avg_action_nll, acc
