from __future__ import annotations
from abc import ABC, abstractmethod
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import torch as to
from torch import optim

from domains.domain import State
from enums import SearchDir
from models import losses
from models.models import PolicyOrHeuristicModel
from search.node import SearchNode
from search.problem import Problem


if TYPE_CHECKING:
    from search.traj import Trajectory


class Agent(ABC):
    def __init__(self, logdir: Path, args: Namespace, aux_args: dict):
        aux_args["bidirectional"] = self.is_bidirectional
        aux_args["has_policy"] = self.has_policy
        aux_args["has_heuristic"] = self.has_heuristic

        self.logdir: Path = logdir
        self.args: Namespace = args
        # todo right model type
        self.model: PolicyOrHeuristicModel = PolicyOrHeuristicModel(args, aux_args)

        if args.mode == "train":
            # todo right optimizer params
            self.optimizer = getattr(optim, args.optimizer)(
                self.model.learnable_params,
            )

            if "mse" in args.loss_fn:
                self.loss_fn = partial(
                    getattr(losses, args.loss_fn), weight=args.weight_mse_loss
                )
            elif "metric" in args.loss_fn:
                self.loss_fn = partial(
                    getattr(losses, args.loss_fn),
                    children_weight=args.children_weight,
                    adj_consistency=args.adj_consistency,
                    adj_weight=args.adj_weight,
                    ends_consistency=args.ends_consistency,
                    ends_weight=args.ends_weight,
                    n_samples=args.n_samples,
                    samples_weight=args.samples_weight,
                )
            else:
                self.loss_fn = getattr(losses, args.loss_fn)

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
        self,
        state: State,
        state_t: to.Tensor,
        actions: list[int],
        mask: to.Tensor,
        forward: bool,
        goal_feats: to.Tensor | None,
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
        open_list: list[SearchNode],  # PQ
        direction: SearchDir,
        children: list[SearchNode],
        children_state_ts: list[to.Tensor],
        masks: list[to.Tensor],
        goal_feats: to.Tensor | None,
    ):
        pass

    @abstractmethod
    def search(
        self, problem: Problem, exp_budget: int, time_budget: float
    ) -> tuple[int, int, Optional[tuple[Trajectory, Optional[Trajectory]]]]:
        pass
