import torch as to
import torch.nn.functional as F


def levin_loss(trajectory, model):
    actions = to.tensor(trajectory.actions)

    states = to.stack(trajectory.states)
    logits = model(states)

    loss = F.cross_entropy(logits, actions)
    loss *= to.tensor(trajectory.num_expanded)

    return loss, logits


def cross_entropy_loss(trajectory, model):
    logits = model(trajectory.states)
    loss = F.cross_entropy(logits, trajectory.actions)

    return loss, logits
