import torch as to
import torch.nn.functional as F
import torch_scatter as ts

from search.utils import MergedTrajectory


def levin_loss_avg(trajs: MergedTrajectory, model):
    logits = model(trajs.states)
    action_nlls = F.cross_entropy(logits, trajs.actions, reduction="none")
    traj_nlls = ts.scatter_mean(action_nlls, trajs.indices, dim=0)
    loss = to.dot(traj_nlls, trajs.nums_expanded) / trajs.num_trajs
    avg_action_nll = action_nlls.detach().mean().item()

    return loss, avg_action_nll, logits.detach()


def levin_loss_sum(trajs: MergedTrajectory, model):
    logits = model(trajs.states)
    action_nlls = F.cross_entropy(logits, trajs.actions, reduction="none")
    traj_nlls = ts.scatter_add(action_nlls, trajs.indices, dim=0)
    loss = to.dot(traj_nlls, trajs.nums_expanded) / trajs.num_trajs
    avg_action_nll = action_nlls.detach().mean().item()

    return loss, avg_action_nll, logits.detach()


def cross_entropy_loss(trajs: MergedTrajectory, model):
    logits = model(trajs.states)
    loss = F.cross_entropy(logits, trajs.actions)
    avg_action_nll = loss.detach().item()

    return loss, avg_action_nll, logits.detach()
