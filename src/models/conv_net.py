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

import torch as to
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class AgentModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.model_args: dict = model_args
        self.num_actions: int = model_args.num_actions
        self.state_t_width: int = model_args.state_t_width
        self.in_channels: int = model_args.in_channels
        self.kernel_size: tuple[int, int] = model_args.kernel_size
        self.num_filters: int = model_args.num_filters
        self.bidirectional: bool = model_args["bidirectional"]
        self.save_path: Path

        self.feature_net: nn.Module = ConvBlock(
            self.in_channels,
            self.state_t_width,
            self.kernel_size,
            self.num_filters,
            self.num_actions,
        )
        self.num_features: int = self.num_filters * ((self.state_t_width - 2) ** 2)

        self.forward_policy: nn.Module = SinglePolicy(
            self.num_features,
            self.num_actions,
            model_args["forward_hidden_layer_sizes"],
        )

        self.bidirectional = model_args["bidirectional"]
        if self.bidirectional:
            self.backward_policy: nn.Module = DoublePolicy(
                self.num_features,
                self.num_actions,
                model_args["backward_hidden_layer_sizes"],
            )

        def save(self, suffix=""):
            tmp_path = self.save_path
            if suffix:
                tmp_path = tmp_path.with_stem(tmp_path.stem + f"_{suffix}")
            to.save(self.state_dict(), self.save_path)


class SinglePolicy(nn.Module):
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
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], num_actions))

        for i in range(len(self.layers) - 1):
            to.nn.init.kaiming_uniform_(
                self.layers[i].weight, mode="fan_in", nonlinearity="relu"
            )
        to.nn.init.xavier_uniform_(self.layers[-1].weight)

    def forward(self, state_feats):
        for i in range(len(self.layers) - 1):
            state_feats = F.relu(self.layers[i](state_feats))

        logits = self.layers[-1](state_feats)

        return logits


class DoublePolicy(nn.Module):
    def __init__(
        self,
        num_features,
        num_actions,
        hidden_layer_sizes=[256, 192, 128],
    ):
        super().__init__()
        self.num_features = num_features * 2
        self.num_actions = num_actions

        self.layers = nn.ModuleList([nn.Linear(num_features, hidden_layer_sizes[0])])
        self.layers.extend(
            [
                nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
                for i in range(len(hidden_layer_sizes) - 1)
            ]
        )
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], num_actions))

        for i in range(len(self.layers) - 1):
            to.nn.init.kaiming_uniform_(
                self.layers[i].weight, mode="fan_in", nonlinearity="relu"
            )
        to.nn.init.xavier_uniform_(self.layers[-1].weight)

    def forward(self, state_feats, goal_feats):
        bs = state_feats.shape[0]
        if goal_feats.shape[0] != bs:
            goal_feats = goal_feats.expand(bs, -1)

        x = to.cat((state_feats, goal_feats), dim=-1)
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))

        logits = self.layers[-1](x)

        return logits


class ConvBlock(nn.Module):
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


class ConvNetSingle(nn.Module):
    def __init__(
        self, in_channels, state_t_width, kernel_size, num_filters, num_actions
    ):
        super().__init__()
        self.num_actions = num_actions
        self.convblock = ConvBlock(
            in_channels, state_t_width, kernel_size, num_filters, num_actions
        )
        self.linear1 = nn.Linear(num_filters * self.convblock.reduced_size, 128)
        to.nn.init.kaiming_uniform_(
            self.linear1.weight, mode="fan_in", nonlinearity="relu"
        )

        self.linear2 = nn.Linear(128, num_actions)
        to.nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        x = self.convblock(x)
        x = F.relu(self.linear1(x))
        logits = self.linear2(x)

        return logits


class ConvNetDouble(nn.Module):
    def __init__(
        self, in_channels, state_t_width, kernel_size, num_filters, num_actions
    ):
        super().__init__()
        self.num_actions = num_actions
        self.convblock1 = ConvBlock(
            in_channels, state_t_width, kernel_size, num_filters, num_actions
        )
        self.convblock2 = ConvBlock(
            in_channels, state_t_width, kernel_size, num_filters, num_actions
        )

        self.linear1 = nn.Linear(num_filters * 2 * self.convblock1.reduced_size, 128)
        to.nn.init.kaiming_uniform_(
            self.linear1.weight, mode="fan_in", nonlinearity="relu"
        )

        self.linear2 = nn.Linear(128, 128)
        to.nn.init.kaiming_uniform_(
            self.linear2.weight, mode="fan_in", nonlinearity="relu"
        )

        self.linear3 = nn.Linear(128, num_actions)
        to.nn.init.xavier_uniform_(self.linear3.weight)

    def forward(self, x):
        """
        x1: current state
        x2: goal state
        """
        x1 = self.convblock1(x[:, 0])
        x2 = self.convblock2(x[:, 1])

        x = to.cat((x1, x2), dim=-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        logits = self.linear3(x)

        return logits
