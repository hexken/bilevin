import torch as to
import math
import torch.nn as nn
import torch.nn.functional as F


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
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size, padding="valid")
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
        self.linear2 = nn.Linear(128, num_actions)

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
        self.linear1 = nn.Linear(num_filters * 2 * self.convblock1.reduced_size, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, num_actions)

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


class HeuristicConvNetSingle(nn.Module):
    def __init__(
        self, in_channels, state_t_width, kernel_size, num_filters, num_actions
    ):
        super().__init__()
        self.num_actions = num_actions
        self.convblock = ConvBlock(
            in_channels, state_t_width, kernel_size, num_filters, num_actions
        )
        self.linear1 = nn.Linear(num_filters * self.convblock.reduced_size, 128)
        self.linear2 = nn.Linear(128, 1)

    def call(self, x):
        x = self.convblock(x)
        x = F.relu(self.linear1(x))
        h = self.linear2(x)

        return h


class TwoHeadedConvNetSingle(nn.Module):
    def __init__(
        self, in_channels, kernel_size, state_t_width, num_filters, num_actions
    ):
        super().__init__()
        self.num_actions = num_actions
        self.convblock = ConvBlock(
            in_channels, state_t_width, kernel_size, num_filters, num_actions
        )

        self.linear_c1 = nn.Linear(num_filters * self.convblock.reduced_size, 128)
        self.linear_c2 = nn.Linear(128, num_actions)

        self.linear_h1 = nn.Linear(num_filters * self.convblock.reduced_size, 128)
        self.linear_h2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.convblock(x)

        logits = F.relu(self.linear_c1(x))
        logits = self.linear_c2(logits)

        h = F.relu(self.linear_h1(x))
        h = self.linear_h2(h)

        return logits, h
