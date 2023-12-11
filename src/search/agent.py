from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import Optional

import torch as to

from domains.domain import State
from enums import TwoDir
from models.models import AgentModel
from search.utils import Problem, SearchNode, Trajectory


class Agent(ABC):
    def __init__(self, logdir: Path, args: Namespace, aux_args: dict):
        aux_args["bidirectional"] = self.is_bidirectional
        aux_args["has_policy"] = self.has_policy
        aux_args["has_heuristic"] = self.has_heuristic

        self.logdir: Path = logdir
        self.model: AgentModel = AgentModel(args, aux_args)

    def save_model(
        self,
        suffix: str = "",
        subpath: str = "",
        log: bool = True,
    ):
        path = self.logdir / subpath / f"model_{suffix}.pt"
        to.save(self.model.state_dict(), path)
        if log:
            print(f"Saved model\n  to {str(path)}")

    @property
    @abstractmethod
    def is_bidirectional(self) -> bool:
        pass

    @property
    @abstractmethod
    def has_policy(self) -> bool:
        pass

    @property
    @abstractmethod
    def has_heuristic(self) -> bool:
        pass

    @abstractmethod
    def make_start_node(
        self, state: State, state_t: to.Tensor, actions: list[int], mask: to.Tensor
    ) -> SearchNode:
        pass

    @abstractmethod
    def make_partial_child_node(
        self,
        parent_node: SearchNode,
        parent_action: int,
        actions: list[int],
        mask: to.Tensor,
        new_state: State,
    ) -> SearchNode:
        pass

    @abstractmethod
    def finalize_children_nodes(
        self,
        open_list: list[SearchNode], # PQ
        direction: TwoDir,
        children: list[SearchNode],
        children_state_ts: list[to.Tensor],
        masks: list[to.Tensor],
        goal_feats: to.Tensor | None,
    ):
        pass

    @abstractmethod
    def search(
        self, problem: Problem, expansion_budget: int, time_budget: float
    ) -> tuple[int, int, tuple[Trajectory, Optional[Trajectory]]]:
        pass
