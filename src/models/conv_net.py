import tensorflow as tf
import torch as to
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from models.loss_functions import (
    LevinLoss,
    CrossEntropyLoss,
    CrossEntropyMSELoss,
    LevinMSELoss,
    MSELoss,
    ImprovedLevinLoss,
    ImprovedLevinMSELoss,
    RegLevinLoss,
    RegLevinMSELoss,
)


class InvalidLossFunction(Exception):
    pass


class HeuristicConvNet(nn.Module):
    def __init__(
        self, in_channels, kernel_size, filters, number_actions, reg_const=0.001
    ):
        super().__init__()
        self._reg_const = reg_const
        self._kernel_size = kernel_size
        self._filters = filters
        self._number_actions = number_actions

        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size, padding="valid")
        self.conv2 = nn.Conv2d(filters, filters, kernel_size, padding="valid")

        self.linear1 = nn.Linear(filters * 3 * 3, 128)
        self.linear2 = nn.Linear(128, 1)

    def predict(self, x):
        return self(x).numpy()

    def call(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.linear1(x.flatten()))
        x = self.linear2(x)

        return x

    def get_number_actions(self):
        return self._number_actions


class TwoHeadedConvNet(tf.keras.Model):
    def __init__(
        self, in_channels, kernel_size, filters, number_actions, reg_const=0.001
    ):
        super().__init__()
        self._reg_const = reg_const
        self._kernel_size = kernel_size
        self._filters = filters
        self._number_actions = number_actions

        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size, padding="valid")
        self.conv2 = nn.Conv2d(filters, filters, kernel_size, padding="valid")

        self.linear_p1 = nn.Linear(filters * 3 * 3, 128)
        self.linear_p2 = nn.Linear(128, number_actions)

        self.linear_h1 = nn.Linear(filters * 3 * 3, 128)
        self.linear_h2 = nn.Linear(128, 1)

    def predict(self, x):
        logits, probs, _, h = self(x)
        return logits.numpy(), probs.numpy(), h.numpy()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten()

        action_logits = F.relu(self.linear_p1(x))
        action_logits = self.linear_p2(action_logits)
        action_log_probs = F.log_softmax(action_logits)
        action_probs = F.softmax(action_logits)

        h = F.relu(self.linear_h1(x))
        h = self.linear_h2(h)

        return action_log_probs, action_probs, action_logits, h

    def get_number_actions(self):
        return self._number_actions


class ConvNet(tf.keras.Model):
    def __init__(
        self, in_channels, kernel_size, filters, number_actions, reg_const=0.001
    ):
        super().__init__()
        self._reg_const = reg_const
        self._kernel_size = kernel_size
        self._filters = filters
        self._number_actions = number_actions

        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size, padding="valid")
        self.conv2 = nn.Conv2d(filters, filters, kernel_size, padding="valid")

        self.linear1 = nn.Linear(filters * 3 * 3, 128)
        self.linear2 = nn.Linear(128, number_actions)

    def predict(self, x):
        log_probs, probs = self(x)
        return log_probs.numpy(), probs.numpy()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten()

        action_logits = F.relu(self.linear1(x))
        action_logits = self.linear2(action_logits)
        action_log_probs = F.log_softmax(action_logits)
        action_probs = F.softmax(action_logits)

        return action_log_probs, action_probs, action_logits

    def _cross_entropy_loss(self, states, y):
        images = [s.get_image_representation() for s in states]
        _, _, logits = self(np.array(images))
        return self.cross_entropy_loss(y, logits)

    def get_number_actions(self):
        return self._number_actions
