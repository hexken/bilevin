from argparse import Namespace
import pickle
from typing import Optional
from functools import partial

import torch as to
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.functional import log_softmax

import models.losses as losses
from models.utils import update_common_params


class SuperModel(nn.Module):
    def __init__(
        self,
        args: Namespace,
        derived_args: dict,
    ):
        super().__init__()

        self.is_bidirectional: bool = derived_args["bidirectional"]

        self.num_actions: int = derived_args["num_actions"]
        self.in_channels: int = derived_args["in_channels"]
        self.state_t_width: int = derived_args["state_t_width"]
        self.state_t_depth: int = derived_args["state_t_depth"]
        self.kernel_size: tuple[int, int] = args.kernel_size
        self.num_kernels: int = args.num_kernels

        self.has_policy: bool = derived_args["has_policy"]
        self.has_heuristic: bool = derived_args["has_heuristic"]

        self.has_feature_net: bool = not args.no_feature_net
        self.share_feature_net: bool = args.share_feature_net
        self.conditional_backward: bool = derived_args["conditional_backward"]

        learnable_params = []
        # create feature net if necessary
        # assumes 2d or 3d tensor of one_hot tensors
        if self.has_feature_net:
            if args.share_feature_net:
                f_feat_lr = b_feat_lr = args.forward_feature_net_lr
            else:
                f_feat_lr = args.forward_feature_net_lr
                b_feat_lr = args.backward_feature_net_lr

            reduced_width: int = self.state_t_width - 2 * self.kernel_size[1] + 2
            if self.kernel_size[0] > 1:
                self.state_t_depth: int = derived_args["state_t_depth"]
                reduced_depth = self.state_t_depth - 2 * self.kernel_size[0] + 2
                self.num_features = (
                    self.num_kernels * reduced_depth * reduced_width**2
                )
            else:
                self.num_features = self.num_kernels * reduced_width**2

            self.forward_feature_net: nn.Module = ConvFeatureNet(
                self.in_channels,
                self.state_t_width,
                self.kernel_size,
                self.num_kernels,
                self.num_actions,
            )
            params = {
                "params": self.forward_feature_net.parameters(),
                "lr": f_feat_lr,
            }
            update_common_params(args, params)
            learnable_params.append(params)

            # todo is it okay to pass shared feature params to optimizer twice?
            if self.is_bidirectional:
                if self.share_feature_net:
                    self.backward_feature_net: nn.Module = self.forward_feature_net
                else:
                    self.backward_feature_net: nn.Module = ConvFeatureNet(
                        self.in_channels,
                        self.state_t_width,
                        self.kernel_size,
                        self.num_kernels,
                        self.num_actions,
                    )
                    params = {
                        "params": self.backward_feature_net.parameters(),
                        "lr": b_feat_lr,
                    }
                    update_common_params(args, params)
                    learnable_params.append(params)
        else:
            # no feature net
            self.num_features = derived_args["num_features"]

        # create policy/herusitic nets
        if self.has_policy:
            self.forward_policy: nn.Module = StatePolicy(
                self.num_features,
                self.num_actions,
                args.forward_policy_layers,
            )
            params = {
                "params": self.forward_policy.parameters(),
                "lr": args.forward_policy_lr,
            }
            update_common_params(args, params)
            learnable_params.append(params)
            if self.is_bidirectional:
                if self.conditional_backward:
                    self.backward_policy: nn.Module = StateGoalPolicy(
                        self.num_features,
                        self.num_actions,
                        args.backward_policy_layers,
                    )
                else:
                    self.backward_policy: nn.Module = StatePolicy(
                        self.num_features,
                        self.num_actions,
                        args.backward_policy_layers,
                    )
                params = {
                    "params": self.backward_policy.parameters(),
                    "lr": args.backward_policy_lr,
                }
                update_common_params(args, params)
                learnable_params.append(params)

        if self.has_heuristic:
            self.forward_heuristic: nn.Module = StateHeuristic(
                self.num_features,
                self.num_actions,
                args.forward_heuristic_layers,
            )
            params = {
                "params": self.forward_heuristic.parameters(),
                "lr": args.forward_heuristic_lr,
            }
            update_common_params(args, params)
            learnable_params.append(params)
            if self.is_bidirectional:
                if self.conditional_backward:
                    self.backward_heuristic: nn.Module = StateGoalHeuristic(
                        self.num_features,
                        self.num_actions,
                        args.backward_heuristic_layers,
                    )
                else:
                    self.backward_heuristic: nn.Module = StateHeuristic(
                        self.num_features,
                        self.num_actions,
                        args.backward_heuristic_layers,
                    )
                params = {
                    "params": self.backward_heuristic.parameters(),
                    "lr": args.backward_heuristic_lr,
                }
                update_common_params(args, params)
                learnable_params.append(params)

        if args.mode == "train":
            self.optimizer = getattr(optim, args.optimizer)(
                learnable_params,
            )
            if "mse" in args.loss_fn:
                self.loss_fn = partial(
                    getattr(losses, args.loss_fn), args.weight_mse_loss
                )
            else:
                self.loss_fn = getattr(losses, args.loss_fn)

        # load model if specified explicitly or from checkpoint
        if args.model_path is not None:
            self.load_state_dict(to.load(args.model_path))
            print(f"Loaded model\n  {str(args.model_path)}")
        elif args.checkpoint_path is not None:
            with args.checkpoint_path.open("rb") as f:
                chkpt_dict = to.load(f)
                self.load_state_dict(chkpt_dict["model_state"])
            print(f"Loaded model\n  {str(args.checkpoint_path)}")

    def forward(
        self,
        state_t: to.Tensor,
        mask: Optional[to.Tensor] = None,
        forward: bool = True,
        goal_feats: Optional[to.Tensor] = None,
        goal_state_t: Optional[to.Tensor] = None,
    ):
        logits = log_probs = h = None
        if forward:
            if self.has_feature_net:
                feats = self.forward_feature_net(state_t)
            else:
                feats = state_t.flatten(start_dim=1)
            if self.has_policy:
                logits = self.forward_policy(feats)
            if self.has_heuristic:
                h = self.forward_heuristic(feats)

        else:
            if self.has_feature_net:
                feats = self.backward_feature_net(state_t)
            else:
                feats = state_t.flatten(start_dim=1)

            # set goal_feats
            if self.conditional_backward:
                if goal_state_t is not None:
                    # goal_feats is None
                    if self.has_feature_net:
                        goal_feats = self.backward_feature_net(goal_state_t)
                    else:
                        goal_feats = goal_state_t.flatten(start_dim=1)

            if self.has_policy:
                if self.conditional_backward:
                    logits = self.backward_policy(feats, goal_feats)
                else:
                    logits = self.backward_policy(feats)

            if self.has_heuristic:
                if self.conditional_backward:
                    h = self.backward_heuristic(feats, goal_feats)
                else:
                    h = self.backward_heuristic(feats)

        if logits is not None:
            if mask is not None:
                # mask[i] should be True if the action i is invalid, False otherwise
                logits = logits.masked_fill(mask, -1e9)
            log_probs = log_softmax(logits, dim=-1)

        return log_probs, h


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

    def forward(self, state_feats: to.Tensor):
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

    def forward(self, state_feats: to.Tensor):
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
        kernel_size: tuple[int, int],
        num_filters: int,
        num_actions: int,
    ):
        super().__init__()
        self._kernel_size: tuple[int, int] = kernel_size
        self._filters: int = num_filters
        self.num_actions: int = num_actions

        if self._kernel_size[0] > 1:
            ks = (*kernel_size, kernel_size[1])
            self.conv1 = nn.Conv3d(
                in_channels,
                num_filters,
                ks,
                padding="valid",
            )
            self.conv2 = nn.Conv3d(
                num_filters,
                num_filters,
                ks,
                padding="valid",
            )
        else:
            ks = (kernel_size[1], kernel_size[1])
            self.conv1 = nn.Conv2d(in_channels, num_filters, ks, padding="valid")
            self.conv2 = nn.Conv2d(num_filters, num_filters, ks, padding="valid")

    def forward(self, x: to.Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.flatten(1)

        return x
