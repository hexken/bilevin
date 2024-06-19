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

from search.traj import Trajectory

if TYPE_CHECKING:
    from models.models import PolicyOrHeuristicModel


def loss_wrapper(
    traj: Trajectory,
    model: PolicyOrHeuristicModel,
    policy_loss=None,
    heuristic_loss=None,
) -> to.Tensor:
    log_probs, hs = model(
        traj.states,
        forward=traj.forward,
        goal_state_t=traj.goal_state_t,
        mask=traj.masks,
    )

    if policy_loss is not None and heuristic_loss is not None:
        p_loss = policy_loss(log_probs, traj)
        h_loss = heuristic_loss(hs, traj)
        loss = p_loss + h_loss
    elif policy_loss is not None:
        loss = policy_loss(log_probs, traj)
    elif heuristic_loss is not None:
        loss = heuristic_loss(hs, traj)

    return loss


def mse(
    hs: to.Tensor,
    traj: Trajectory,
):
    loss = mse_loss(hs, traj.cost_to_gos.unsqueeze(1))

    return loss


def nll(
    log_probs: to.Tensor,
    traj: Trajectory,
    levin=False,
    reduction="sum",
):
    loss = nll_loss(log_probs, traj.actions, reduction=reduction)
    if levin:
        loss = loss * traj.num_expanded

    return loss
