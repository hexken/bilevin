# Copyright (C) 2021-2022, Ken Tjhia
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch as to
import torch.distributed as dist

from models import AgentModel
import models.losses as losses
import search.utils as sutils
from search.utils import Problem, SearchNode, Trajectory


class Agent(ABC):
    def __init__(self, rank, logdir, args, model_args):
        if not self.trainable:
            return

        model_args["bidirectioal"] = self.bidirectional

        self.logdir: Path = logdir

        self.model: AgentModel
        self.cost_fn: Callable[[SearchNode], float] = getattr(sutils, args.cost_fn)

        if args.model_path is None:
            # just use the random initialization from rank 0
            self.model = AgentModel(model_args)
            if args.world_size > 1:
                for param in self.model.parameters():
                    dist.broadcast(param.data, 0)
        elif args.model_path.is_dir():
            full_model_path = args.model_path / f"best_{args.model_suffix}.pt"
            self.model = to.load(full_model_path)
            if rank == 0:
                print(f"Loaded model\n  {str(full_model_path)}")
        else:
            raise ValueError("model-path argument must be a directory if given")

        if rank == 0:
            init_model = self.logdir / f"model_init.pt"
            to.save(self.model, init_model)
            print(f"Saved init model\n  to {str(init_model)}")

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
        to.save(self.model, path)
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
    def search(
        self, problem: Problem, expansion_budget: int, time_budget: int
    ) -> tuple[int, int, int, int, tuple[Trajectory, Optional[Trajectory]]]:
        pass
