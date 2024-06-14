from __future__ import annotations
from math import ceil
from typing import TYPE_CHECKING

import numpy as np
import torch as to
from torch import clamp, exp, log, sum
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.nn.functional import log_softmax, nll_loss, normalize
from torch.nn.functional import cross_entropy as cross_entropy_loss

from search.traj import MetricTrajectory, Trajectory, BYOLTrajectory

if TYPE_CHECKING:
    from models.models import PolicyOrHeuristicModel, BYOL


def mse(traj: Trajectory, model: PolicyOrHeuristicModel):
    _, h = model(
        traj.states,
        forward=traj.forward,
        goal_state_t=traj.goal_state_t,
        mask=traj.masks,
    )
    loss = mse_loss(h, traj.cost_to_gos.unsqueeze(1))

    return loss, 0.0, 0.0


def phs(traj: Trajectory, model: PolicyOrHeuristicModel, policy_loss):
    p_loss, avg_action_nll, acc = policy_loss(traj, model)
    mse_loss, _, _ = mse(traj, model)
    loss = p_loss + mse_loss

    return loss, avg_action_nll, acc


def traj_nll_sum(traj: Trajectory, model: PolicyOrHeuristicModel):
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


def traj_nll_avg(traj: Trajectory, model: PolicyOrHeuristicModel):
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

    return loss, avg_action_nll, acc


def levin_sum(traj: Trajectory, model: PolicyOrHeuristicModel):
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


def levin_avg(traj: Trajectory, model: PolicyOrHeuristicModel):
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
