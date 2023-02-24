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
import torch.nn.functional as F
import torch_scatter as ts

from search.utils import MergedTrajectory


def levin_loss_avg(trajs: MergedTrajectory, model):
    logits = model(trajs.states)
    action_nlls = F.cross_entropy(logits, trajs.actions, reduction="none")
    traj_nlls = ts.scatter_mean(action_nlls, trajs.indices, dim=0)
    loss = to.dot(traj_nlls, trajs.nums_expanded) / trajs.num_trajs
    avg_action_nll = action_nlls.detach().mean().item()

    return loss, avg_action_nll, logits.detach()


def levin_loss(trajs: MergedTrajectory, model):
    logits = model(trajs.states)
    action_nlls = F.cross_entropy(logits, trajs.actions, reduction="none")
    traj_nlls = ts.scatter_add(action_nlls, trajs.indices, dim=0)
    loss = to.dot(traj_nlls, trajs.nums_expanded) / trajs.num_trajs
    avg_action_nll = action_nlls.detach().mean().item()

    return loss, avg_action_nll, logits.detach()


def cross_entropy_loss(trajs: MergedTrajectory, model):
    logits = model(trajs.states)
    loss = F.cross_entropy(logits, trajs.actions)
    avg_action_nll = loss.detach().item()

    return loss, avg_action_nll, logits.detach()
