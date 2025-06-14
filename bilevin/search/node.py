from __future__ import annotations
from typing import Generic, Optional, TYPE_CHECKING

from torch import Tensor

from enums import SearchDir

from domains.state import TState

if TYPE_CHECKING:
    from domains.domain import Domain


class DirStructures:
    def __init__(
        self,
        dir: SearchDir,
        open: list[SearchNode],
        closed: dict[SearchNode, SearchNode],
        domain: Domain,
        other_domain: Domain,
        goal_feats: Optional[Tensor] = None,
        mask_invalid_actions: bool = False,
    ):
        self.dir = dir
        self.open = open
        self.closed = closed
        self.domain = domain
        self.other_domain = other_domain
        self.goal_feats = goal_feats

        self.expanded = 0
        if mask_invalid_actions:
            self.masks = []
        else:
            self.masks = None
        self.children_to_be_evaluated = []
        self.state_t_of_children_to_be_evaluated = []
        if dir == SearchDir.FORWARD:
            self.next_dir = SearchDir.BACKWARD
        else:
            self.next_dir = SearchDir.FORWARD
        self.next_ds: DirStructures


class SearchNode(Generic[TState]):
    def __init__(
        self,
        state: TState,
        parent: SearchNode | None,
        parent_action: int | None,  # action taken from parent to reach this node
        actions: list[int],
        mask: Tensor | None,
        g: int,
        log_prob: float,
        log_action_probs: Tensor | None = None,
        h: float | None = None,
        f: float = 0,
        ds: DirStructures | None = None,
    ):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.g = g
        self.log_prob = log_prob
        self.actions = actions
        self.mask = mask
        self.log_action_probs = log_action_probs
        self.h = h
        self.f = f
        self.ds = ds = ds

    def __eq__(self, other):
        """
        Verify if two SearchNodes are identical by verifying the
         state in the nodes.
        """
        return self.state == other.state

    def __lt__(self, other):
        """
        less-than used by the heap
        """
        return self.f < other.f

    def __hash__(self):
        """
        Hash function used in the closed list
        """
        return self.state.__hash__()
