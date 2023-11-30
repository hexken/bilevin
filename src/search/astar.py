from math import log

import torch as to

from domains.domain import State
from enums import TwoDir
from search.agent import Agent
from search.bidir import BiDir
from search.unidir import UniDir
from search.utils import SearchNode


class AStarBase(Agent):
    def __init__(self, logdir, args, model_args):
        super().__init__(logdir, args, model_args)

    @property
    def has_policy(self):
        return False

    @property
    def has_heuristic(self):
        return True

    def make_start_node(
        self: Agent,
        state: State,
        state_t: to.Tensor,
        actions: list[int],
        mask: to.Tensor,
    ) -> SearchNode:
        _, _, h = self.model(state_t, mask=mask)

        h = h.item()
        start_node = SearchNode(
            state,
            parent=None,
            parent_action=None,
            actions=actions,
            actions_mask=mask,
            g=0,
            log_prob=0.0,
            f=h,
            h=h,
            log_action_probs=None,
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
        if parent_node.action_hs is None:
            raise ValueError("Parent node has child herusitics")
        g = parent_node.g + 1
        h = parent_node.action_hs[parent_action].item()
        new_node = SearchNode(
            new_state,
            parent=parent_node,
            parent_action=parent_action,
            actions=actions,
            actions_mask=mask,
            g=g,
            h=h,
            f=g + h,
            log_prob=0.0,
        )
        return new_node

    def finalize_children_nodes(
        self: Agent,
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


class AStar(UniDir, AStarBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BiAStar(BiDir, AStarBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
