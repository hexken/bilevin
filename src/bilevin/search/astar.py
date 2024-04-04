from __future__ import annotations
from heapq import heappush
from math import log
from typing import TYPE_CHECKING

import torch as to

from domains.domain import State
from enums import SearchDir
from search.bidir_alt import BiDirAlt
from search.bidir_bfs import BiDirBFS
from search.node import SearchNode
from search.unidir import UniDir

from search.agent import Agent

# if TYPE_CHECKING:


class AStarBase(Agent):
    def __init__(self, logdir, args, model_args):
        if args.loss_fn == "default":
            args.loss_fn = "mse"
        super().__init__(logdir, args, model_args)
        self.w = args.weight_astar

    @property
    def has_policy(self):
        return False

    @property
    def has_heuristic(self):
        return True

    def make_start_node(
        self,
        state: State,
        state_t: to.Tensor,
        actions: list[int],
        mask: to.Tensor,
        forward: bool,
        goal_feats: to.Tensor | None,
    ) -> SearchNode:
        _, h = self.model(state_t, forward=forward, goal_feats=goal_feats)

        h = h.item()
        start_node = SearchNode(
            state,
            parent=None,
            parent_action=None,
            actions=actions,
            actions_mask=mask,
            log_prob=0.0,
            log_action_probs=None,
            g=0,
            h=h,
            f=self.w * h,
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
        new_node = SearchNode(
            new_state,
            parent=parent_node,
            parent_action=parent_action,
            actions=actions,
            actions_mask=mask,
            g=parent_node.g + 1,
            log_prob=0.0,
        )
        return new_node

    def finalize_children_nodes(
        self,
        open_list: list[SearchNode],  # pq
        direction: SearchDir,
        children: list[SearchNode],
        children_state_ts: list[to.Tensor],
        masks: list[to.Tensor],
        goal_feats: to.Tensor | None,
    ):
        children_state_t = to.stack(children_state_ts)
        _, hs = self.model(
            children_state_t,
            forward=direction == SearchDir.FORWARD,
            goal_feats=goal_feats,
        )

        for child, h in zip(children, hs):
            h = h.item()
            child.h = h
            child.f = child.g + self.w * h
            heappush(open_list, child)


class AStar(UniDir, AStarBase):
    def __init__(self, logdir, args, aux_args):
        super().__init__(logdir, args, aux_args)


class BiAStarBFS(BiDirBFS, AStarBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BiAStarAlt(BiDirAlt, AStarBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
