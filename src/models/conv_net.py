import torch as to
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
