import torch as to
import torch.nn as nn
from torch.nn.functional import cross_entropy, log_softmax, nll_loss

from models import AgentModel
from search.utils import Trajectory


def levin_loss_mid(traj: Trajectory, model: AgentModel):
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
