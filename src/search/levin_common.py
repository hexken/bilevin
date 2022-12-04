import torch as to
import math
from .utils import SearchNode


class LevinNode(SearchNode):
    def __init__(
        self,
        state,
        parent=None,
        action=None,
        g_cost=None,
        log_prob=None,
        levin_cost=None,
        log_action_probs=None,
        num_expanded_when_generated=None,
    ):
        super().__init__(state, parent, action, g_cost)
        self.log_prob = log_prob
        self.levin_cost = levin_cost
        self.log_action_probs = log_action_probs

    def __lt__(self, other):
        """
        used by the heap
        """
        return self.levin_cost < other.levin_cost


def levin_cost(node: LevinNode):
    return math.log(node.g_cost + 1) - node.log_prob


def levin_cost_pred_h(node, predicted_h):
    if predicted_h < 0:
        predicted_h = 0
    return math.log(predicted_h + node.g_cost) - node.log_prob
