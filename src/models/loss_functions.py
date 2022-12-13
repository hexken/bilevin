import torch as to
import torch_scatter as ts
import torch.nn.functional as F
from search.utils import MergedTrajectories


def levin_loss_avg(trajs: MergedTrajectories, model):
    logits = model(trajs.states)
    action_nlls = F.cross_entropy(logits, trajs.actions, reduction="none")
    traj_nlls = ts.scatter_mean(action_nlls, trajs.indices, dim=0)
    loss = to.dot(traj_nlls, trajs.nums_expanded) / len(trajs)
    with to.no_grad():
        avg_action_nll = action_nlls.mean()

    return loss, avg_action_nll, logits


def levin_loss_sum(trajs: MergedTrajectories, model):
    logits = model(trajs.states)
    action_nlls = F.cross_entropy(logits, trajs.actions, reduction="none")
    traj_nlls = ts.scatter_add(action_nlls, trajs.indices, dim=0)
    loss = to.dot(traj_nlls, trajs.nums_expanded) / len(trajs)
    with to.no_grad():
        avg_action_nll = action_nlls.mean()

    return loss, avg_action_nll, logits


def cross_entropy_loss(trajs: MergedTrajectories, model):
    logits = model(trajs.states)
    loss = F.cross_entropy(logits, trajs.actions)
    avg_action_nll = loss

    return loss, avg_action_nll, logits
