from argparse import Namespace
from typing import Optional

import timeit
import torch as to
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from torch.nn.functional import log_softmax


class PolicyOrHeuristicModel(nn.Module):
    def __init__(
        self,
        args: Namespace,
        derived_args: dict,
    ):
        super().__init__()

        self.has_policy: bool = derived_args["has_policy"]
        self.has_heuristic: bool = derived_args["has_heuristic"]
        self.is_bidirectional: bool = derived_args["bidirectional"]

        self.has_feature_net: bool = not args.no_feature_net
        self.share_feature_net: bool = args.share_feature_net
        self.conditional_backward: bool = derived_args["conditional_backward"]
        self.conditional_forward: bool = derived_args["conditional_forward"]

        num_actions: int = derived_args["num_actions"]
        in_channels: int = derived_args["in_channels"]
        state_t_width: int = derived_args["state_t_width"]
        state_t_depth: int = derived_args["state_t_depth"]
        kernel_size: tuple[int, int] = args.kernel_size

        learnable_params = []
        # create feature net if necessary
        # assumes 2d or 3d tensor of one_hot tensors
        if self.has_feature_net:
            if args.share_feature_net:
                f_feat_lr = b_feat_lr = args.forward_feature_net_lr
            else:
                f_feat_lr = args.forward_feature_net_lr
                b_feat_lr = args.backward_feature_net_lr

            if args.feature_net_type == "conv":
                self.forward_feature_net: nn.Module = CNN(
                    in_channels,
                    kernel_size,
                    args.n_kernels,
                )
                num_features = get_num_features_after_cnn(
                    kernel_size,
                    state_t_width,
                    state_t_depth,
                    args.n_kernels,
                )
            else:  # linear feature net
                self.forward_feature_net: nn.Module = MLP(
                    derived_args["num_raw_features"],
                    args.forward_feature_layers,
                    args.n_embed_dim,
                )
                num_features = args.n_embed_dim
            params = {
                "params": self.forward_feature_net.parameters(),
                "lr": f_feat_lr,
            }
            learnable_params.append(params)

            if self.is_bidirectional:
                if self.share_feature_net:
                    self.backward_feature_net: nn.Module = self.forward_feature_net
                else:
                    if args.feature_net_type == "conv":
                        self.backward_feature_net: nn.Module = CNN(
                            in_channels,
                            kernel_size,
                            args.n_kernels,
                        )
                    else:
                        self.backward_feature_net: nn.Module = MLP(
                            derived_args["num_raw_features"],
                            args.backward_feature_layers,
                            args.n_embed_dim,
                        )
                    params = {
                        "params": self.backward_feature_net.parameters(),
                        "lr": b_feat_lr,
                    }
                    learnable_params.append(params)
        else:
            # no feature net
            num_features = derived_args["num_raw_features"]

        if self.conditional_forward:
            num_f_features = 2 * num_features
        else:
            num_f_features = num_features
        if self.conditional_backward:
            num_b_features = 2 * num_features
        else:
            num_b_features = num_features

        # create policy/herusitic nets
        if self.has_policy:
            self.forward_policy: nn.Module = MLP(
                num_f_features,
                args.forward_policy_layers,
                num_actions,
            )
            params = {
                "params": self.forward_policy.parameters(),
                "lr": args.forward_policy_lr,
            }
            learnable_params.append(params)
            if self.is_bidirectional:
                self.backward_policy: nn.Module = MLP(
                    num_b_features,
                    args.backward_policy_layers,
                    num_actions,
                )
                params = {
                    "params": self.backward_policy.parameters(),
                    "lr": args.backward_policy_lr,
                }
                learnable_params.append(params)

        if self.has_heuristic:
            self.forward_heuristic: nn.Module = MLP(
                num_f_features,
                args.forward_heuristic_layers,
                1,
            )
            params = {
                "params": self.forward_heuristic.parameters(),
                "lr": args.forward_heuristic_lr,
            }
            learnable_params.append(params)
            if self.is_bidirectional:
                self.backward_heuristic: nn.Module = MLP(
                    num_b_features, args.backward_heuristic_layers, 1
                )
                params = {
                    "params": self.backward_heuristic.parameters(),
                    "lr": args.backward_heuristic_lr,
                }
                learnable_params.append(params)

        self.learnable_params = learnable_params

        # init weights
        for p in self.parameters():
            if p.dim() > 1:
                to.nn.init.xavier_uniform_(p)
            else:
                to.nn.init.constant_(p, 0)

    def forward(
        self,
        state_t: to.Tensor,
        mask: Optional[to.Tensor] = None,
        forward: bool = True,
        goal_feats: Optional[to.Tensor] = None,
        goal_state_t: Optional[to.Tensor] = None,
    ):
        # set goal_feats if necessary
        if goal_state_t is not None:
            # goal_feats is None
            if self.has_feature_net:
                if self.forward:
                    goal_feats = self.forward_feature_net(goal_state_t)
                else:
                    goal_feats = self.backward_feature_net(goal_state_t)
            else:
                goal_feats = goal_state_t.flatten(start_dim=1)

        logits = log_probs = h = None
        if forward:
            if self.has_feature_net:
                feats = self.forward_feature_net(state_t)
            else:
                feats = state_t.flatten(start_dim=1)

            if self.has_policy:
                logits = self.forward_policy(feats, goal_feats)
            if self.has_heuristic:
                h = self.forward_heuristic(feats, goal_feats)

        else:  # backward preds
            if self.has_feature_net:
                feats = self.backward_feature_net(state_t)
            else:
                feats = state_t.flatten(start_dim=1)

            if self.has_policy:
                logits = self.backward_policy(feats, goal_feats)

            if self.has_heuristic:
                h = self.backward_heuristic(feats, goal_feats)

        if logits is not None:
            if mask is not None:
                # mask[i] should be True if the action i is invalid, False otherwise
                logits = logits.masked_fill(mask, -1e9)
            log_probs = log_softmax(logits, dim=-1)

        return log_probs, h


class MLP(nn.Module):
    def __init__(
        self,
        in_size: int,
        hidden_layer_sizes: list[int],
        out_size: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_size, hidden_layer_sizes[0])])
        self.layers.extend(
            [
                nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
                for i in range(len(hidden_layer_sizes) - 1)
            ]
        )
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], out_size)

    def forward(self, x1: to.Tensor, x2: Optional[to.Tensor] = None):
        if x2 is not None:
            x2 = x2.expand(x1.shape[0], -1)
            x1 = to.cat((x1, x2), dim=-1)

        for l in self.layers:
            x1 = F.relu(l(x1))
        out = self.output_layer(x1)
        return out


class CNN(nn.Module):
    """Assumes a square sized input with state_t_width side length"""

    def __init__(
        self,
        in_channels: int,
        kernel_size: tuple[int, int],
        num_filters: int,
    ):
        super().__init__()
        if kernel_size[0] > 1:
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


def get_num_features_after_cnn(
    kernel_size: tuple[int, int],
    state_t_width: int,
    state_t_depth: int,
    num_kernels: int,
):
    reduced_width: int = state_t_width - 2 * kernel_size[1] + 2
    if kernel_size[0] > 1:
        reduced_depth = state_t_depth - 2 * kernel_size[0] + 2
        num_features = num_kernels * reduced_depth * reduced_width**2
    else:
        num_features = num_kernels * reduced_width**2
    return num_features
