import torch as to
import torch_scatter as ts
import torch.nn.functional as F
from search.utils import MergedTrajectories


def levin_loss(trajs: MergedTrajectories, model):
    logits = model(trajs.states)
    action_nlls = F.cross_entropy(logits, trajs.actions, reduction="none")
    traj_nlls = ts.scatter_add(action_nlls, trajs.indices, dim=0)
    loss = to.dot(traj_nlls, trajs.nums_expanded) / len(trajs)

    return loss, logits


def cross_entropy_loss(trajs: MergedTrajectories, model):
    logits = model(trajs.states)
    loss = F.cross_entropy(logits, trajs.actions)

    return loss, logits
