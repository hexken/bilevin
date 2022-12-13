import torch as to
import torch.nn.functional as F
from search.utils import MergedTrajectories


def levin_loss(trajs: MergedTrajectories, model):
    logits = model(trajs.states)
    nlls = F.cross_entropy(logits, trajs.actions, reduction="none")
    loss = to.dot(nlls, trajs.nums_expanded)

    return loss, logits


def cross_entropy_loss(trajs: MergedTrajectories, model):
    logits = model(trajs.states)
    loss = F.cross_entropy(logits, trajs.actions)

    return loss, logits
