import torch as to
import torch.nn as nn
import torch.nn.functional as F

# todo will have to parameterize network sizes to account for all domains


class HeuristicConvNet(nn.Module):
    def __init__(self, in_channels, kernel_size, num_filters, num_actions):
        super().__init__()
        self._kernel_size = kernel_size
        self._filters = num_filters
        self._num_actions = num_actions

        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size, padding="valid")
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size, padding="valid")

        self.linear1 = nn.Linear(num_filters * 3 * 3, 128)
        self.linear2 = nn.Linear(128, 1)

    def call(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        dims = x.shape
        if dims == 4:
            x = x.flatten(1)
        elif dims == 3:
            x = x.flatten()
        else:
            raise ValueError("Invalid input shape")

        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x


class TwoHeadedConvNet(nn.Module):
    def __init__(self, in_channels, kernel_size, num_filters, num_actions):
        super().__init__()
        self._kernel_size = kernel_size
        self._filters = num_filters
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size, padding="valid")
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size, padding="valid")

        self.linear_p1 = nn.Linear(num_filters * 3 * 3, 128)
        self.linear_p2 = nn.Linear(128, num_actions)

        self.linear_h1 = nn.Linear(num_filters * 3 * 3, 128)
        self.linear_h2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        dims = x.shape
        if dims == 4:
            x = x.flatten(1)
        elif dims == 3:
            x = x.flatten()
        else:
            raise ValueError("Invalid input shape")

        logits = F.relu(self.linear_p1(x))
        logits = self.linear_p2(logits)
        # log_probs = F.log_softmax(logits)

        h = F.relu(self.linear_h1(x))
        h = self.linear_h2(h)

        return logits, h


class ConvNet(nn.Module):
    def __init__(self, in_channels, kernel_size, num_filters, num_actions):
        super().__init__()
        self._kernel_size = kernel_size
        self._filters = num_filters
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size, padding="valid")
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size, padding="valid")

        self.linear1 = nn.Linear(num_filters * 3 * 3, 128)
        self.linear2 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        dims = x.shape
        if dims == 4:
            x = x.flatten(1)
        elif dims == 3:
            x = x.flatten()
        else:
            raise ValueError("Invalid input shape")

        logits = F.relu(self.linear1(x))
        logits = self.linear2(logits)
        # log_probs = F.log_softmax(logits)

        return logits
