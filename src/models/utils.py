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
from torch.nn.functional import cross_entropy, nll_loss
import torch_scatter as ts

from models import AgentModel
from search.utils import MergedTrajectory, Trajectory


def levin_loss(trajs: list[Trajectory], model: AgentModel):
    loss = 0
    acc = 0
    total_actions = 0
    avg_action_nll = 0
    for t in trajs:
        log_probs, _ = model(
            t.states, forward=t.forward, goal_state_t=t.goal_state_t, mask=t.masks
        )
        nll = nll_loss(log_probs, t.actions, reduction="sum")
        loss += nll * t.num_expanded

        avg_action_nll += nll.item()
        acc += (log_probs.detach().argmax(dim=1) == t.actions).sum().item()
        total_actions += len(t)

    loss /= len(trajs)
    avg_action_nll /= total_actions
    acc /= total_actions

    return loss, avg_action_nll, acc
