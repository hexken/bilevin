from heapq import heappush
from math import log
from typing import TYPE_CHECKING

import torch as to

from enums import SearchDir
from search.agent import Agent
from search.agents.bidir import BiDir
from search.agents.unidir import UniDir
from search.node import SearchNode
from domains.state import State

if TYPE_CHECKING:
    pass


class LevinBase(Agent):
    def __init__(self, logdir, args, model_args):
        if args.loss_fn == "default":
            args.loss_fn = "levin_avg"
        super().__init__(logdir, args, model_args)

    @property
    def has_policy(self):
        return True

    @property
    def has_heuristic(self):
        return False

    def make_start_node(
        self: Agent,
        state: State,
        state_t: to.Tensor,
        actions: list[int],
        mask: to.Tensor | None,
        forward: bool,
        goal_feats: to.Tensor | None,
    ) -> SearchNode:
        log_probs, _ = self.model(
            state_t, mask=mask, forward=forward, goal_feats=goal_feats
        )

        start_node = SearchNode(
            state,
            parent=None,
            parent_action=None,
            actions=actions,
            mask=mask,
            g=0,
            log_prob=0.0,
            f=0.0,
            log_action_probs=log_probs[0],
        )
        return start_node

    def make_partial_child_node(
        self: Agent,
        parent_node: SearchNode,
        parent_action: int,
        actions: list[int],
        mask: to.Tensor | None,
        new_state: State,
    ) -> SearchNode:
        assert parent_node.log_action_probs is not None
        assert parent_node.log_prob is not None

        g = parent_node.g + 1
        log_prob = (
            parent_node.log_prob + parent_node.log_action_probs[parent_action].item()
        )
        new_node = SearchNode(
            new_state,
            parent=parent_node,
            parent_action=parent_action,
            actions=actions,
            mask=mask,
            log_prob=log_prob,
            g=g,
            f=log(g) - log_prob,
        )
        return new_node

    def finalize_children_nodes(
        self: Agent,
        open_list: list[SearchNode],
        direction: SearchDir,
        children: list[SearchNode],
        children_state_ts: list[to.Tensor],
        masks: list[to.Tensor] | None,
        goal_feats: to.Tensor | None,
    ):
        children_state_t = to.stack(children_state_ts)
        if masks is None:
            masks_t = None
        else:
            masks_t = to.stack(masks)
        log_probs, _ = self.model(
            children_state_t,
            forward=direction == SearchDir.FORWARD,
            goal_feats=goal_feats,
            mask=masks_t,
        )

        for child, lap in zip(children, log_probs):
            child.log_action_probs = lap
            heappush(open_list, child)


class Levin(UniDir, LevinBase):
    def __init__(self, logdir, args, aux_args):
        super().__init__(logdir, args, aux_args)


class BiLevinBFS(BiDir, LevinBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, alternating=False)


class BiLevinAlt(BiDir, LevinBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, alternating=True)


BiLevin = BiLevinAlt
