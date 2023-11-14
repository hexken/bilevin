from __future__ import annotations
import heapq
from timeit import default_timer as timer

import torch as to

from domains.domain import State
from enums import TwoDir
from search.agent import Agent
from search.bidir import Bidir
from search.utils import Problem, SearchNode


class BiAstar(Bidir):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def trainable(cls):
        return True

    def get_start_node(self, state, state_t, actions, mask):
        log_probs, _ = self.model(state_t, mask=mask)

        start_node = SearchNode(
            state,
            parent=None,
            parent_action=None,
            actions=actions,
            actions_mask=mask,
            g_cost=0,
            log_prob=0.0,
            cost=0.0,
            log_action_probs=log_probs[0] if log_probs is not None else None,
        )
        return start_node

    def get_child_node(
        self,
        parent_node: SearchNode,
        parent_action: int,
        actions: list[int],
        mask: to.Tensor,
        new_state: State,
        cost_fn,
    ):
        new_node = SearchNode(
            new_state,
            g_cost=parent_node.g_cost + 1,
            parent=parent_node,
            parent_action=parent_action,
            actions=actions,
            actions_mask=mask,
            log_prob=parent_node.log_prob
            + parent_node.log_action_probs[parent_action].item(),
        )
        new_node.cost = cost_fn(new_node)
        return new_node

    def evaluate_children(
        self,
        direction: TwoDir,
        children: list[SearchNode],
        children_state_ts: list[to.Tensor],
        masks: list[to.Tensor],
        goal_feats: to.Tensor | None,
    ):
        children_state_t = to.stack(children_state_ts)
        masks_t = to.stack(masks)
        log_probs, _ = self.model(
            children_state_t,
            forward=direction == TwoDir.FORWARD,
            goal_feats=goal_feats,
            mask=masks_t,
        )

        for child, lap in zip(children, log_probs):
            child.log_action_probs = lap
