from math import ceil
from typing import TYPE_CHECKING

import torch as to
import torch.nn as nn
from torch.nn.functional import cross_entropy, log_softmax, nll_loss
from torch.nn.functional import mse_loss as mse

from search.traj import Trajectory

if TYPE_CHECKING:
    from models.models import SuperModel


def mse_loss(traj: Trajectory, model: "SuperModel"):
    _, h = model(
        traj.states,
        forward=traj.forward,
        goal_state_t=traj.goal_state_t,
        mask=traj.masks,
    )
    loss = mse(h, traj.cost_to_gos.unsqueeze(1))

    return loss, 0.0, 0.0


def cross_entropy_loss(traj: Trajectory, model: "SuperModel"):
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


def cross_entropy_mse_loss(traj: Trajectory, model: "SuperModel"):
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
    loss = ce_loss + mse_loss

    return loss, avg_action_nll, acc


def cross_entropy_mid_loss(traj: Trajectory, model: "SuperModel"):
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


def levin_loss(traj: Trajectory, model: "SuperModel"):
    n_actions = len(traj)
    log_probs, _ = model(
        traj.states,
        forward=traj.forward,
        goal_state_t=traj.goal_state_t,
        mask=traj.masks,
    )
    nll = nll_loss(log_probs, traj.actions, reduction="sum")
    loss = nll * traj.num_expanded

    avg_action_nll = nll.item() / n_actions
    acc = (log_probs.detach().argmax(dim=1) == traj.actions).sum().item() / n_actions

    return loss, avg_action_nll, acc
