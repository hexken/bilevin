from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

import torch as to
import torch.nn as nn
from torch.nn.functional import cross_entropy, log_softmax, nll_loss
from torch.nn.functional import mse_loss as mse

from search.traj import MetricTrajectory, Trajectory

if TYPE_CHECKING:
    from models.models import SuperModel


def metric_loss(
    f_traj: MetricTrajectory, b_traj: MetricTrajectory, model: SuperModel, weight=1.0
):
    loss = 0.0
    n = len(f_traj)
    # compute all necessary features
    f_states_feats = model(f_traj.states)
    b_states_feats = model(b_traj.states, forward=False)

    f_children_feats = [model(children) for children in f_traj.children]
    b_children_feats = [model(children, forward=False) for children in b_traj.children]

    for feats, children_feats in zip(f_states_feats, f_children_feats):
        sqnorm = (children_feats - feats).pow(2).sum()
        loss += sqnorm
        loss += weight * (sqnorm - 1.0) ** 2

    for feats, children_feats in zip(b_states_feats, b_children_feats):
        sqnorm = (children_feats - feats).pow(2).sum()
        loss += sqnorm
        loss += weight * (sqnorm - 1.0) ** 2

    # forward/backward adjacents
    for i in range(len(f_traj.states) - 1):
        sqnorm = (f_states_feats[i] - b_states_feats[n - i - 2]).pow(2).sum()
        loss += sqnorm
        loss += weight * (sqnorm - 1.0) ** 2

    return loss


def mse_loss(traj: Trajectory, model: SuperModel, weight=1.0):
    _, h = model(
        traj.states,
        forward=traj.forward,
        goal_state_t=traj.goal_state_t,
        mask=traj.masks,
    )
    loss = mse(h, traj.cost_to_gos.unsqueeze(1))

    return loss, 0.0, 0.0


def cross_entropy_loss(traj: Trajectory, model: SuperModel):
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


def cross_entropy_mse_loss(traj: Trajectory, model: SuperModel, weight=1.0):
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


def levin_avg_mse_loss(traj: Trajectory, model: SuperModel, weight=1.0):
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


def traj_nll_mse_loss(traj: Trajectory, model: SuperModel, weight=1.0):
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


def levin_sum_mse_loss(traj: Trajectory, model: SuperModel, weight=1.0):
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


def levin_avg_loss(traj: Trajectory, model: SuperModel):
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


def traj_nll_loss(traj: Trajectory, model: SuperModel):
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


def levin_sum_loss(traj: Trajectory, model: SuperModel):
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


def cross_entropy_mid_loss(traj: Trajectory, model: SuperModel):
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
