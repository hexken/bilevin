from heapq import heappush
from math import log
from typing import TYPE_CHECKING

import torch as to

from domains.domain import State
from enums import SearchDir
from search.agent import Agent
from search.agents.bidir_alt import BiDirAlt
from search.agents.bidir_bfs import BiDirBFS
from search.agents.unidir import UniDir
from search.node import SearchNode

if TYPE_CHECKING:
    pass


class PHSBase(Agent):
    def __init__(self, logdir, args, model_args):
        if args.loss_fn == "default":
            args.loss_fn = "levin_avg"
        super().__init__(logdir, args, model_args)

    @property
    def has_policy(self):
        return True

    @property
    def has_heuristic(self):
        return True

    def make_start_node(
        self: Agent,
        state: State,
        state_t: to.Tensor,
        actions: list[int],
        mask: to.Tensor,
        forward: bool,
        goal_feats: to.Tensor | None,
    ) -> SearchNode:
        log_probs, h = self.model(
            state_t, mask=mask, forward=forward, goal_feats=goal_feats
        )

        h = h.item()
        if h < 0:
            h = 0
        start_node = SearchNode(
            state,
            parent=None,
            parent_action=None,
            actions=actions,
            actions_mask=mask,
            g=0,
            log_prob=0.0,
            f=log(1 + h),
            log_action_probs=log_probs[0],
        )
        return start_node

    def make_partial_child_node(
        self: Agent,
        parent_node: SearchNode,
        parent_action: int,
        actions: list[int],
        mask: to.Tensor,
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
            actions_mask=mask,
            log_prob=log_prob,
            g=g,
        )
        return new_node

    def finalize_children_nodes(
        self: Agent,
        open_list: list[SearchNode],
        direction: SearchDir,
        children: list[SearchNode],
        children_state_ts: list[to.Tensor],
        masks: list[to.Tensor],
        goal_feats: to.Tensor | None,
    ):
        children_state_t = to.stack(children_state_ts)
        masks_t = to.stack(masks)
        log_probs, hs = self.model(
            children_state_t,
            forward=direction == SearchDir.FORWARD,
            goal_feats=goal_feats,
            mask=masks_t,
        )

        for child, lap, h in zip(children, log_probs, hs):
            h = h.item()
            if h < 0:
                h = 0
            # pg = child.g + 1
            pg = child.g
            child.log_action_probs = lap
            child.h = h
            child.f = log(pg + h) - (1 + (h / pg)) * child.log_prob
            heappush(open_list, child)


class PHS(UniDir, PHSBase):
    def __init__(self, logdir, args, aux_args):
        super().__init__(logdir, args, aux_args)


class BiPHSBFS(BiDirBFS, PHSBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BiPHSAlt(BiDirAlt, PHSBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


BiPHS = BiPHSAlt
