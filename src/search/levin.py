from heapq import heappush
from math import log

import torch as to

from domains.domain import State
from enums import TwoDir
from search.agent import Agent
from search.bidir_bfs import BiDirBFS
from search.bidir_alt import BiDirAlt
from search.unidir import UniDir
from search.utils import SearchNode


class LevinBase(Agent):
    def __init__(self, logdir, args, model_args):
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
        mask: to.Tensor,
        forward: bool,
        goal_feats: to.Tensor | None,
    ) -> SearchNode:
        log_probs, _, _ = self.model(
            state_t, mask=mask, forward=forward, goal_feats=goal_feats
        )

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
            f=log(g) - log_prob,
        )
        return new_node

    def finalize_children_nodes(
        self: Agent,
        open_list: list[SearchNode],
        direction: TwoDir,
        children: list[SearchNode],
        children_state_ts: list[to.Tensor],
        masks: list[to.Tensor],
        goal_feats: to.Tensor | None,
    ):
        children_state_t = to.stack(children_state_ts)
        masks_t = to.stack(masks)
        log_probs, _, _ = self.model(
            children_state_t,
            forward=direction == TwoDir.FORWARD,
            goal_feats=goal_feats,
            mask=masks_t,
        )

        for child, lap in zip(children, log_probs):
            child.log_action_probs = lap
            heappush(open_list, child)


class Levin(UniDir, LevinBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BiLevinBFS(BiDirBFS, LevinBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class BiLevinAlt(BiDirAlt, LevinBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
