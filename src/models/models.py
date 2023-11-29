from argparse import Namespace
import pickle
from typing import Optional

import torch as to
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax

import models.losses as losses


# todo this module is poorly organized
class AgentModel(nn.Module):
    def __init__(
        self,
        args: Namespace,
        model_args: dict,
    ):
        super().__init__()
        self.is_bidirectional: bool = model_args["bidirectional"]
        self.model_args: dict = model_args
        self.num_actions: int = model_args["num_actions"]

        self.share_feature_net: bool = model_args["share_feature_net"]
        self.conditional_backward: bool = model_args["conditional_backward"]

        self.in_channels: int = model_args["in_channels"]
        self.num_filters: int = model_args["num_filters"]
        self.kernel_size: tuple[int, int] = model_args["kernel_size"]
        self.kernel_depth = None

        self.state_t_width: int = model_args["state_t_width"]

        # create feature net if necessary
        if not args.no_feature_net:
            reduced_width: int = self.state_t_width - 2 * self.kernel_size[0] + 2
            # particularly hacky
            if model_args["kernel_depth"] > 1:
                self.state_t_depth: int = model_args["state_t_depth"]
                self.kernel_depth = model_args["kernel_depth"]
                self.kernel_size = (self.kernel_depth, *self.kernel_size)
                reduced_depth = self.state_t_depth - 2 * self.kernel_depth + 2
                self.num_features = (
                    self.num_filters * reduced_depth * reduced_width**2
                )
            else:
                self.num_features = self.num_filters * reduced_width**2

            self.forward_feature_net: nn.Module = ConvFeatureNet(
                self.in_channels,
                self.state_t_width,
                self.kernel_size,
                self.num_filters,
                self.num_actions,
            )

            if self.is_bidirectional:
                if self.share_feature_net:
                    self.backward_feature_net: nn.Module = self.forward_feature_net
                else:
                    self.backward_feature_net: nn.Module = ConvFeatureNet(
                        self.in_channels,
                        self.state_t_width,
                        self.kernel_size,
                        self.num_filters,
                        self.num_actions,
                    )

            # create policy/herusitic nets
            if model_args["has_policy"]:
                self.forward_policy: nn.Module = StatePolicy(
                    self.num_features,
                    self.num_actions,
                    model_args["forward_policy_layers"],
                )
                if self.bidirectional:
                    if self.conditional_backward:
                        self.backward_policy: nn.Module = StateGoalPolicy(
                            self.num_features,
                            self.num_actions,
                            model_args["backward_policy_layers"],
                        )
                    else:
                        self.backward_policy: nn.Module = StatePolicy(
                            self.num_features,
                            self.num_actions,
                            model_args["backward_policy_layers"],
                        )

            if model_args["has_heuristic"]:
                self.forward_heuristic: nn.Module = StateHeuristic(
                    self.num_features,
                    self.num_actions,
                    model_args["forward_heuristic_layers"],
                )
                if self.bidirectional:
                    if self.conditional_backward:
                        self.backward_policy: nn.Module = StateGoalHeuristic(
                            self.num_features,
                            self.num_actions,
                            model_args["backward_heuristic_layers"],
                        )
                    else:
                        self.backward_policy: nn.Module = StatePolicy(
                            self.num_features,
                            self.num_actions,
                            model_args["backward_heuristic_layers"],
                        )

        # load model if specified explicitly or from checkpoint
        if args.model_path is not None:
            self.load_state_dict(to.load(args.model_path))
            print(f"Loaded model\n  {str(args.model_path)}")
        elif args.checkpoint_path is not None:
            with args.checkpoint_path.open("rb") as f:
                chkpt_dict = pickle.load(f)
                self.load_state_dict(chkpt_dict["model_state"])
            print(f"Loaded model\n  {str(args.checkpoint_path)}")

        # Set up the optimizer
        if args.mode == "train":
            self.loss_fn = getattr(losses, args.loss_fn)
            if not args.share_feature_net:
                flr = args.forward_feature_net_lr
                blr = args.backward_feature_net_lr
            else:
                flr = args.feature_net_lr
                blr = args.feature_net_lr

            optimizer_params = [
                {
                    "params": self.forward_feature_net.parameters(),
                    "lr": flr,
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": self.forward_policy.parameters(),
                    "lr": args.forward_policy_lr,
                    "weight_decay": args.weight_decay,
                },
            ]
            if self.is_bidirectional:
                if not args.share_feature_net:
                    optimizer_params.append(
                        {
                            "params": self.backward_feature_net.parameters(),
                            "lr": blr,
                            "weight_decay": args.weight_decay,
                        }
                    )
                optimizer_params.append(
                    {
                        "params": self.backward_policy.parameters(),
                        "lr": args.backward_policy_lr,
                        "weight_decay": args.weight_decay,
                    }
                )
            self.optimizer = to.optim.Adam(optimizer_params)

    def forward(
        self,
        state_t: to.Tensor,
        mask: Optional[to.Tensor] = None,
        forward: bool = True,
        goal_feats: Optional[to.Tensor] = None,
        goal_state_t: Optional[to.Tensor] = None,
    ):
        if forward:
            if self.forward_feature_net is not None:
                feats = self.forward_feature_net(state_t)
            else:
                feats = state_t
            logits = self.forward_policy(feats)
        else:
            if self.backward_feature_net is not None:
                feats = self.backward_feature_net(state_t)
            else:
                feats = state_t
            feats = self.backward_feature_net(state_t)
            if goal_feats is not None:
                logits = self.backward_policy(feats, goal_feats)
            elif goal_state_t is not None:
                goal_feats = self.backward_feature_net(goal_state_t)
                logits = self.backward_policy(feats, goal_feats)
            else:
                logits = self.backward_policy(feats)

        if mask is not None:
            # mask[i] should be True if the action i is invalid, False otherwise
            logits = logits.masked_fill(mask, -1e9)

        log_probs = log_softmax(logits, dim=-1)

        return log_probs, logits

    def get_feats(self, state_t: to.Tensor, forward: bool = True):
        if forward:
            feats = self.forward_feature_net(state_t)
        else:
            feats = self.backward_feature_net(state_t)

        return feats


class StateHeuristic(nn.Module):
    def __init__(
        self, num_features: int, num_actions: int, hidden_layer_sizes: list[int] = [128]
    ):
        super().__init__()
        self.num_features: int = num_features
        self.num_actions: int = num_actions

        self.layers = nn.ModuleList([nn.Linear(num_features, hidden_layer_sizes[0])])
        self.layers.extend(
            [
                nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
                for i in range(len(hidden_layer_sizes) - 1)
            ]
        )
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], 1)

    def forward(self, state_feats: to.Tensor, goal_feats: Optional[to.Tensor] = None):
        for l in self.layers:
            state_feats = F.relu(l(state_feats))

        h = self.output_layer(state_feats)

        return h


class StateGoalHeuristic(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_actions: int,
        hidden_layer_sizes=[256, 192, 128],
    ):
        super().__init__()
        self.num_features: int = num_features * 2
        self.num_actions: int = num_actions

        self.layers = nn.ModuleList(
            [nn.Linear(self.num_features, hidden_layer_sizes[0])]
        )
        self.layers.extend(
            [
                nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
                for i in range(len(hidden_layer_sizes) - 1)
            ]
        )
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], 1)

    def forward(self, state_feats: to.Tensor, goal_feats: to.Tensor):
        bs = state_feats.shape[0]
        if goal_feats.shape[0] != bs:
            goal_feats = goal_feats.expand(bs, -1)

        x = to.cat((state_feats, goal_feats), dim=-1)
        for l in self.layers:
            x = F.relu(l(x))

        h = self.output_layer(x)

        return h


class StatePolicy(nn.Module):
    def __init__(
        self, num_features: int, num_actions: int, hidden_layer_sizes: list[int] = [128]
    ):
        super().__init__()
        self.num_features: int = num_features
        self.num_actions: int = num_actions

        self.layers = nn.ModuleList([nn.Linear(num_features, hidden_layer_sizes[0])])
        self.layers.extend(
            [
                nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
                for i in range(len(hidden_layer_sizes) - 1)
            ]
        )
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], num_actions)

    def forward(self, state_feats: to.Tensor, goal_feats: Optional[to.Tensor] = None):
        for l in self.layers:
            state_feats = F.relu(l(state_feats))

        logits = self.output_layer(state_feats)

        return logits


class StateGoalPolicy(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_actions: int,
        hidden_layer_sizes=[256, 192, 128],
    ):
        super().__init__()
        self.num_features: int = num_features * 2
        self.num_actions: int = num_actions

        self.layers = nn.ModuleList(
            [nn.Linear(self.num_features, hidden_layer_sizes[0])]
        )
        self.layers.extend(
            [
                nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
                for i in range(len(hidden_layer_sizes) - 1)
            ]
        )
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], num_actions)

    def forward(self, state_feats: to.Tensor, goal_feats: to.Tensor):
        bs = state_feats.shape[0]
        if goal_feats.shape[0] != bs:
            goal_feats = goal_feats.expand(bs, -1)

        x = to.cat((state_feats, goal_feats), dim=-1)
        for l in self.layers:
            x = F.relu(l(x))

        logits = self.output_layer(x)

        return logits


class ConvFeatureNet(nn.Module):
    """Assumes a square sized input with state_t_width side length"""

    def __init__(
        self,
        in_channels: int,
        state_t_width: int,
        kernel_size: tuple[int, int] | tuple[int, int, int],
        num_filters: int,
        num_actions: int,
    ):
        super().__init__()
        self._kernel_size: tuple[int, int] = kernel_size
        self._filters: int = num_filters
        self.num_actions: int = num_actions

        if len(kernel_size) == 3:
            self.conv1 = nn.Conv3d(
                in_channels, num_filters, kernel_size, padding="valid"
            )
            self.conv2 = nn.Conv3d(
                num_filters, num_filters, kernel_size, padding="valid"
            )
        elif len(kernel_size) == 2:
            self.conv1 = nn.Conv2d(
                in_channels, num_filters, kernel_size, padding="valid"
            )
            self.conv2 = nn.Conv2d(
                num_filters, num_filters, kernel_size, padding="valid"
            )
        else:
            raise ValueError("kernel_size must be 2 or 3 dimensional")

        self.reduced_size = (state_t_width - 2) ** 2

    def forward(self, x: to.Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.flatten(1)

        return x
