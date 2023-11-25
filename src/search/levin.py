from __future__ import annotations
from math import log

import torch as to

from domains.domain import State
from enums import TwoDir
from search.agent import Agent
from search.bidir import BiDir
from search.udir import UniDir
from search.utils import SearchNode


def _get_start_node(
    self: Agent,
    state: State,
    state_t: to.Tensor,
    actions: list[int],
    mask: to.Tensor,
) -> SearchNode:
    log_probs, _ = self.model(state_t, mask=mask)

    start_node = SearchNode(
        state,
        parent=None,
        parent_action=None,
        actions=actions,
        actions_mask=mask,
        g=0,
        log_prob=0.0,
        f=0.0,
        log_action_probs=log_probs[0],
    )
    return start_node


def _get_child_node(
    self: Agent,
    parent_node: SearchNode,
    parent_action: int,
    actions: list[int],
    mask: to.Tensor,
    new_state: State,
) -> SearchNode:
    if parent_node.log_action_probs is None:
        raise ValueError("Parent node has no log_action_probs")
    elif parent_node.log_prob is None:
        raise ValueError("Parent node has no log_prob")
    new_node = SearchNode(
        new_state,
        g=parent_node.g + 1,
        parent=parent_node,
        parent_action=parent_action,
        actions=actions,
        actions_mask=mask,
        log_prob=parent_node.log_prob
        + parent_node.log_action_probs[parent_action].item(),
    )
    new_node.f = log(new_node.g) - new_node.log_prob
    return new_node


def _evaluate_children(
    self: Agent,
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


class Levin(UniDir):
    @property
    def trainable(cls):
        return True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_start_node(
        self, state: State, state_t: to.Tensor, actions: list[int], mask: to.Tensor
    ) -> SearchNode:
        return _get_start_node(self, state, state_t, actions, mask)

    def get_child_node(
        self,
        parent_node: SearchNode,
        parent_action: int,
        actions: list[int],
        mask: to.Tensor,
        new_state: State,
    ) -> SearchNode:
        return _get_child_node(
            self, parent_node, parent_action, actions, mask, new_state
        )

    def evaluate_children(
        self,
        direction: TwoDir,
        children: list[SearchNode],
        children_state_ts: list[to.Tensor],
        masks: list[to.Tensor],
        goal_feats: to.Tensor | None,
    ):
        _evaluate_children(
            self, direction, children, children_state_ts, masks, goal_feats
        )


class BiLevin(BiDir):
    @property
    def trainable(cls):
        return True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_start_node(
        self, state: State, state_t: to.Tensor, actions: list[int], mask: to.Tensor
    ) -> SearchNode:
        return _get_start_node(self, state, state_t, actions, mask)

    def get_child_node(
        self,
        parent_node: SearchNode,
        parent_action: int,
        actions: list[int],
        mask: to.Tensor,
        new_state: State,
    ) -> SearchNode:
        return _get_child_node(
            self, parent_node, parent_action, actions, mask, new_state
        )

    def evaluate_children(
        self,
        direction: TwoDir,
        children: list[SearchNode],
        children_state_ts: list[to.Tensor],
        masks: list[to.Tensor],
        goal_feats: to.Tensor | None,
    ):
        _evaluate_children(
            self, direction, children, children_state_ts, masks, goal_feats
        )
