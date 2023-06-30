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

from models import AgentModel
from search.utils import Trajectory


def loop_levin_loss(trajs: list[Trajectory], model: AgentModel):
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


def merge_cross_entropy(trajs: list[Trajectory], model: AgentModel):
    n_trajs = len(trajs)
    forward = trajs[0].forward

    merged_states = to.cat(tuple(t.states for t in trajs))
    merged_actions = to.cat(tuple(t.actions for t in trajs))
    merged_masks = to.cat(tuple(t.masks for t in trajs))

    lengths = to.tensor(tuple(len(t) for t in trajs))
    sum_indices = to.repeat_interleave(to.arange(n_trajs), lengths)

    if not forward:
        repeats = to.tensor(tuple(len(t) for t in trajs))
        goal_states_t = to.cat(tuple(t.goal_state_t for t in trajs))
        merged_goal_states_t = to.repeat_interleave(
            goal_states_t, repeats=repeats, dim=0
        )
    else:
        merged_goal_states_t = None

    log_probs, _ = model(
        merged_states,
        forward=forward,
        goal_state_t=merged_goal_states_t,
        mask=merged_masks,
    )
    loss = nll_loss(log_probs, merged_actions, reduction="mean")
    avg_action_nll = loss.detach()
    acc = (
        (log_probs.detach().argmax(dim=1) == merged_actions)
        .sum()
        .div(len(log_probs))
        .item()
    )

    return loss, avg_action_nll, acc
