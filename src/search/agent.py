from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch as to
import torch.distributed as dist
from models import AgentModel
import models.losses as losses
import search.utils as sutils
from search.utils import Problem, SearchNode, Trajectory


# todo separate some things to model

class Agent(ABC):
    def __init__(self, logdir, args, model_args):
        # todo why this here?
        if not self.trainable:
            return

        model_args["bidirectional"] = self.bidirectional

        self.logdir: Path = logdir

        self.model: AgentModel
        self.cost_fn: Callable[[SearchNode], float] = getattr(sutils, args.cost_fn)

        self.model = AgentModel(model_args)
        if args.model_path is not None:
            self.model.load_state_dict(to.load(args.model_path))
            print(f"Loaded model\n  {str(args.model_path)}")

        # Set up the optimizer and loss_fn
        if args.mode == "train":
            assert self.model
            self.loss_fn = getattr(losses, args.loss_fn)
            if not args.share_feature_net:
                flr = args.forward_feature_net_lr
                blr = args.backward_feature_net_lr
            else:
                flr = args.feature_net_lr
                blr = args.feature_net_lr

            optimizer_params = [
                {
                    "params": self.model.forward_feature_net.parameters(),
                    "lr": flr,
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": self.model.forward_policy.parameters(),
                    "lr": args.forward_policy_lr,
                    "weight_decay": args.weight_decay,
                },
            ]
            if self.bidirectional:
                if not args.share_feature_net:
                    optimizer_params.append(
                        {
                            "params": self.model.backward_feature_net.parameters(),
                            "lr": blr,
                            "weight_decay": args.weight_decay,
                        }
                    )
                optimizer_params.append(
                    {
                        "params": self.model.backward_policy.parameters(),
                        "lr": args.backward_policy_lr,
                        "weight_decay": args.weight_decay,
                    }
                )
            self.optimizer = to.optim.Adam(optimizer_params)

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
    @classmethod
    @abstractmethod
    def bidirectional(cls) -> bool:
        pass

    @property
    @classmethod
    @abstractmethod
    def trainable(cls) -> bool:
        pass

    @abstractmethod
    def get_start_node(self, *args, **kwargs) -> SearchNode:
        pass

    @abstractmethod
    def get_child_node(self, *args, **kwargs) -> SearchNode:
        pass

    @abstractmethod
    def evaluate_children(self, *args, **kwargs):
        pass

    @abstractmethod
    def search(
        self, problem: Problem, expansion_budget: int, time_budget: int
    ) -> tuple[int, int, tuple[Trajectory, Optional[Trajectory]]]:
        pass
