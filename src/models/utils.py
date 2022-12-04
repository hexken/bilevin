import torch as to
import torch.nn.functional as F


def mixture_uniform(logits, weight_uniform=0.01):
    probs = to.exp(F.log_softmax(logits, dim=0))
    log_probs = to.log((1 - weight_uniform) * probs + weight_uniform * (1 / len(probs)))
    return log_probs
