import torch as to
import torch.nn.functional as F


def mixture_uniform(logits: to.Tensor, weight_uniform: float = 0):
    if weight_uniform == 0:
        return to.log_softmax(logits, dim=-1)
    else:
        probs = to.exp(F.log_softmax(logits, dim=-1))
        log_probs = to.log(
            (1 - weight_uniform) * probs + weight_uniform * (1 / len(probs))
        )
        return log_probs
