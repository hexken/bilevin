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

from pathlib import Path
from typing import Optional

import torch as to
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax


class AgentModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.bidirectional: bool = model_args["bidirectional"]
        self.model_args: dict = model_args
        self.in_channels: int = model_args["in_channels"]
        self.kernel_size: tuple[int, int] = model_args["kernel_size"]
        self.num_filters: int = model_args["num_filters"]
        self.share_feature_net: bool = model_args["share_feature_net"]

        self.num_features: int = self.num_filters * ((self.state_t_width - 2) ** 2)
        self.requires_backward_goal: bool = model_args["requires_backward_goal"]
        self.state_t_width: int = model_args["state_t_width"]
        self.num_actions: int = model_args["num_actions"]

        self.forward_policy: nn.Module = StatePolicy(
            self.num_features,
            self.num_actions,
            model_args["forward_hidden_layers"],
        )

        self.forward_feature_net: nn.Module = ConvFeatureNet(
            self.in_channels,
            self.state_t_width,
            self.kernel_size,
            self.num_filters,
            self.num_actions,
        )

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

        if self.bidirectional:
            if self.requires_backward_goal:
                self.backward_policy: nn.Module = StateGoalPolicy(
                    self.num_features,
                    self.num_actions,
                    model_args["backward_hidden_layers"],
                )
            else:
                self.backward_policy: nn.Module = StatePolicy(
                    self.num_features,
                    self.num_actions,
                    model_args["backward_hidden_layers"],
                )

    def forward(
        self, state_t, mask=None, forward=None, goal_feats=None, goal_state_t=None
    ):
        if forward:
            feats = self.forward_feature_net(state_t)
            logits = self.forward_policy(feats)
        else:
            feats = self.backward_feature_net(state_t)
            if goal_feats:
                logits = self.backward_policy(feats, goal_feats)
            elif goal_state_t:
                goal_feats = self.backward_feature_net(goal_state_t)
                logits = self.backward_policy(feats, goal_feats)
            else:
                logits = self.backward_policy(feats)

        if mask:
            # mask[i] should be True if the action i is invalid, False otherwise
            logits = logits.masked_fill(mask, -1e9)

        log_probs = log_softmax(logits)

        return log_probs, logits

    def get_feats(self, state_t: to.Tensor, forward=True):
        if forward:
            feats = self.forward_feature_net(state_t)
        else:
            feats = self.backward_feature_net(state_t)

        return feats


class StatePolicy(nn.Module):
    def __init__(self, num_features, num_actions, hidden_layer_sizes=[128]):
        super().__init__()
        self.num_features = num_features
        self.num_actions = num_actions

        self.layers = nn.ModuleList([nn.Linear(num_features, hidden_layer_sizes[0])])
        self.layers.extend(
            [
                nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
                for i in range(len(hidden_layer_sizes) - 1)
            ]
        )
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], num_actions)

        for i in range(len(self.layers)):
            to.nn.init.kaiming_uniform_(
                self.layers[i].weight, mode="fan_in", nonlinearity="relu"
            )
            to.nn.init.constant_(self.layers[i].bias, 0.0)
        to.nn.init.xavier_uniform_(self.output_layer.weight)
        to.nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, state_feats):
        for l in self.layers:
            state_feats = F.relu(l(state_feats))

        logits = self.output_layer(state_feats)

        return logits


class StateGoalPolicy(nn.Module):
    def __init__(
        self,
        num_features,
        num_actions,
        hidden_layer_sizes=[256, 192, 128],
    ):
        super().__init__()
        self.num_features = num_features * 2
        self.num_actions = num_actions

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

        for i in range(len(self.layers)):
            to.nn.init.kaiming_uniform_(
                self.layers[i].weight, mode="fan_in", nonlinearity="relu"
            )
            to.nn.init.constant_(self.layers[i].bias, 0.0)
        to.nn.init.xavier_uniform_(self.output_layer.weight)
        to.nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, state_feats, goal_feats):
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
        self, in_channels, state_t_width, kernel_size, num_filters, num_actions
    ):
        super().__init__()
        self._kernel_size = kernel_size
        self._filters = num_filters
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size, padding="valid")
        to.nn.init.kaiming_uniform_(
            self.conv1.weight, mode="fan_in", nonlinearity="relu"
        )

        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size, padding="valid")
        to.nn.init.kaiming_uniform_(
            self.conv2.weight, mode="fan_in", nonlinearity="relu"
        )

        self.reduced_size = (state_t_width - 2) ** 2

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.flatten(1)

        return x
