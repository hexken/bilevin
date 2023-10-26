import torch as to
import torch.nn as nn
from math import floor
from torch.nn.functional import cross_entropy, log_softmax, nll_loss

from models import AgentModel
from search.utils import Trajectory


def cross_entropy_avg_loss(traj: Trajectory, model: AgentModel):
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


def cross_entropy_sum_loss(traj: Trajectory, model: AgentModel):
    n_actions = len(traj)
    log_probs, _ = model(
        traj.states,
        forward=traj.forward,
        goal_state_t=traj.goal_state_t,
        mask=traj.masks,
    )
    loss = nll_loss(log_probs, traj.actions, reduction="sum")

    avg_action_nll = loss.item() / n_actions
    acc = (log_probs.detach().argmax(dim=1) == traj.actions).sum().item() / n_actions

    return loss, avg_action_nll, acc


def cross_entropy_mid_loss(traj: Trajectory, model: AgentModel):
    n_actions = len(traj)
    print(traj.states.shape)
    mp = floor(n_actions / 2)
    goal_state_t = traj.states[mp]
    states = traj.states[:mp]
    print(states.shape)
    actions = traj.actions[:mp]
    masks = traj.masks[:mp]

    log_probs, _ = model(
        states,
        forward=traj.forward,
        goal_state_t=goal_state_t,
        mask=masks,
    )
    loss = nll_loss(log_probs, actions, reduction="sum")

    avg_action_nll = loss.item() / n_actions
    acc = (log_probs.detach().argmax(dim=1) == actions).sum().item() / n_actions

    return loss, avg_action_nll, acc


def levin_loss(traj: Trajectory, model: AgentModel):
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
