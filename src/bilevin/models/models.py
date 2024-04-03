from argparse import Namespace
from typing import Optional

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

        self.is_bidirectional: bool = derived_args["bidirectional"]

        self.has_policy: bool = derived_args["has_policy"]
        self.has_heuristic: bool = derived_args["has_heuristic"]

        self.has_feature_net: bool = not args.no_feature_net
        self.share_feature_net: bool = args.share_feature_net
        self.conditional_backward: bool = derived_args["conditional_backward"]

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
                num_raw_features = derived_args["num_features"]
                self.forward_feature_net: nn.Module = MLP(
                    num_raw_features, args.forward_feature_layers, args.n_embed_dim
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
                            num_features,
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
            num_features = derived_args["num_features"]

        # create policy/herusitic nets
        if self.has_policy:
            self.forward_policy: nn.Module = MLP(
                num_features,
                args.forward_policy_layers,
                num_actions,
            )
            params = {
                "params": self.forward_policy.parameters(),
                "lr": args.forward_policy_lr,
            }
            learnable_params.append(params)
            if self.is_bidirectional:
                if self.conditional_backward:
                    self.backward_policy: nn.Module = MLP(
                        2 * num_features,
                        args.backward_policy_layers,
                        num_actions,
                    )
                else:
                    self.backward_policy: nn.Module = MLP(
                        num_features,
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
                num_features,
                args.forward_heuristic_layers,
                1,
            )
            params = {
                "params": self.forward_heuristic.parameters(),
                "lr": args.forward_heuristic_lr,
            }
            learnable_params.append(params)
            if self.is_bidirectional:
                if self.conditional_backward:
                    self.backward_heuristic: nn.Module = MLP(
                        2 * num_features, args.backward_heuristic_layers, 1
                    )
                else:
                    self.backward_heuristic: nn.Module = MLP(
                        num_features, args.backward_heuristic_layers, 1
                    )
                params = {
                    "params": self.backward_heuristic.parameters(),
                    "lr": args.backward_heuristic_lr,
                }
                learnable_params.append(params)

        self.learnable_params = learnable_params
        load_model(args, self)

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

        if log_probs is not None or h is not None:
            return log_probs, h
        else:
            return feats


class BYOL(nn.Module):
    def __init__(
        self,
        args: Namespace,
        derived_args: dict,
        tau: float = 0.99,
    ):
        super().__init__()
        self.online: nn.Module = _BYOL(args, derived_args)
        self.learnable_params = [
            {"params": self.online.parameters(), "lr": args.forward_feature_net_lr}
        ]
        self.target: nn.Module = _BYOL(args, derived_args, make_predictor=False)
        self.tau = tau

    def update_target(self):
        # todo no predictor in target
        online_params = chain(
            self.online.encoder.parameters(), self.online.projector.parameters()
        )
        target_params = chain(
            self.target.encoder.parameters(), self.target.projector.parameters()
        )
        for online, target in zip(online_params, target_params):
            target.data = self.tau * target.data + (1 - self.tau) * online.data

    def forward(
        self,
        state1_t: to.Tensor,
        state2_t: to.Tensor,
    ):
        states = to.cat((state1_t, state2_t), dim=0)
        _, online_preds = self.online(states, for_loss=True)
        online_pred1, online_pred2 = to.chunk(online_preds, 2, dim=0)

        with to.no_grad():
            target_projs, _ = self.target(states, for_loss=True)
            target_proj1, target_proj2 = to.chunk(target_projs, 2, dim=0)

        # todo may need to detach target_projs
        return online_pred1, online_pred2, target_proj1, target_proj2


class _BYOL(nn.Module):
    def __init__(
        self,
        args: Namespace,
        derived_args: dict,
        make_predictor: bool = True,
    ):
        super().__init__()

        in_channels: int = derived_args["in_channels"]
        state_t_width: int = derived_args["state_t_width"]
        state_t_depth: int = derived_args["state_t_depth"]
        kernel_size: tuple[int, int] = args.kernel_size

        num_features = get_num_features_after_cnn(
            kernel_size,
            state_t_width,
            state_t_depth,
            args.n_kernels,
        )
        layer_sizes = [2 * num_features]

        self.encoder: nn.Module = CNN(
            in_channels,
            kernel_size,
            args.n_kernels,
        )
        self.projector: nn.Module = MLP(num_features, layer_sizes, args.n_embed_dim)

        self.predictor: Optional[nn.Module]
        if make_predictor:
            self.predictor = MLP(args.n_embed_dim, layer_sizes, args.n_embed_dim)
        else:
            self.predictor = None

    def forward(
        self,
        state_t: to.Tensor,
        for_loss: bool = False,
    ):
        x = self.encoder(state_t)
        if for_loss:
            y = self.projector(x)
            if self.predictor is not None:
                y = self.predictor(y)
            else:
                y = None
            return x, y
        else:
            return x


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


def load_model(args: Namespace, model: nn.Module):
    if args.model_path is not None:
        model.load_state_dict(to.load(args.model_path))
        print(f"Loaded model\n  {str(args.model_path)}")
    elif args.checkpoint_path is not None:
        with args.checkpoint_path.open("rb") as f:
            chkpt_dict = to.load(f)
            model.load_state_dict(chkpt_dict["model_state"])
        print(f"Loaded model\n  {str(args.checkpoint_path)}")


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
