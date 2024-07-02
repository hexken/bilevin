from __future__ import annotations
from abc import ABC, abstractmethod
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import torch as to
from torch import optim

from enums import SearchDir
from models.losses import loss_wrapper, mse, nll
from models.models import PolicyOrHeuristicModel
from search.node import SearchNode


if TYPE_CHECKING:
    from loaders import Problem
    from search.traj import Trajectory
    from domains.domain import State


class Agent(ABC):
    def __init__(self, logdir: Path, args: Namespace, aux_args: dict):
        aux_args["bidirectional"] = self.is_bidirectional
        aux_args["has_policy"] = self.has_policy
        aux_args["has_heuristic"] = self.has_heuristic

        self.logdir: Path = logdir
        self.args: Namespace = args

        if "avg" in args.loss_fn:
            reduction = "mean"
        else:
            reduction = "sum"
        if "levin" in args.loss_fn:
            levin = True
        else:
            levin = False

        if "PHS" in args.agent:
            policy_loss = partial(nll, reduction=reduction, levin=levin)
            self.loss_fn = partial(
                loss_wrapper, policy_loss=policy_loss, heuristic_loss=mse
            )
        elif "AStar" in args.agent:
            self.loss_fn = partial(loss_wrapper, policy_loss=None, heuristic_loss=mse)
        elif "Levin" in args.agent:
            policy_loss = partial(nll, reduction=reduction, levin=levin)
            self.loss_fn = partial(
                loss_wrapper, policy_loss=policy_loss, heuristic_loss=None
            )
        else:
            raise ValueError(f"Unknown agent {args.agent}")

        self.model: PolicyOrHeuristicModel = PolicyOrHeuristicModel(args, aux_args)
        self.load_model(args)
        self.optimizer = getattr(optim, args.optimizer)(
            self.model.learnable_params, weight_decay=args.weight_decay
        )

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

    def load_model(self, args: Namespace):
        if args.model_path is not None:
            self.model.load_state_dict(to.load(args.model_path))
            print(f"Loaded model\n  {str(args.model_path)}")
        elif args.checkpoint_path is not None:
            with args.checkpoint_path.open("rb") as f:
                chkpt_dict = to.load(f)
                self.model.load_state_dict(chkpt_dict["model_state"])
            print(f"Loaded model\n  {str(args.checkpoint_path)}")

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
