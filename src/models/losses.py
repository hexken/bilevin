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

from typing import Optional

import torch as to
import torch.nn as nn
from torch.nn.functional import cross_entropy, log_softmax, nll_loss

from models import AgentModel
from search.utils import Trajectory


def loop_levin_loss_real(
    trajs: list[Trajectory], model: AgentModel
):
    loss = 0
    acc = 0
    total_actions = 0
    avg_action_nll = 0
    for t in trajs:
        log_probs, _ = model(
            t.states, forward=t.forward, goal_state_t=t.goal_state_t, mask=t.masks
        )
        nll = nll_loss(log_probs, t.actions, reduction="sum")
        loss += nll * (t.cost / to.exp(-1 * nll))

        avg_action_nll += nll.item()
        acc += (log_probs.detach().argmax(dim=1) == t.actions).sum().item()
        total_actions += len(t)

    loss /= len(trajs)
    avg_action_nll /= total_actions
    acc /= total_actions

    return loss, avg_action_nll, acc


def loop_levin_loss(trajs: list[Trajectory], model: AgentModel):
    loss = 0
    acc = 0
    total_actions = 0
    avg_action_nll = 0
    forward = trajs[0].forward
    for t in trajs:
        log_probs, _ = model(
            t.states, forward=forward, goal_state_t=t.goal_state_t, mask=t.masks
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


def loop_cross_entropy_loss(
    trajs: list[Trajectory], model: AgentModel, n_subgoals: int = 0
):
    loss = 0
    acc = 0
    total_actions = 0
    avg_action_nll = 0
    forward = trajs[0].forward

    for t in trajs:
        log_probs, _ = model(
            t.states, forward=forward, goal_state_t=t.goal_state_t, mask=t.masks
        )
        nll = nll_loss(log_probs, t.actions, reduction="sum")
        loss += nll
        avg_action_nll += nll.item()
        acc += (log_probs.detach().argmax(dim=1) == t.actions).sum().item()
        total_actions += len(t)

    loss /= total_actions
    avg_action_nll /= total_actions
    acc /= total_actions

    return loss, avg_action_nll, acc


# def loop_cross_entropy_loss(
#     trajs: list[Trajectory], model: AgentModel, n_subgoals: int = 0
# ):
#     """
#     It's messy but we need speed here, I think looping and repeating calculations is quicker than
#     creating the batches and calling functions.
#     """
#     loss = 0
#     acc = 0
#     avg_action_nll = 0
#     total_actions = 0
#     forward = trajs[0].forward
#     for t in trajs:
#         t_len = len(t)
#         total_actions += t_len
#         if forward:
#             log_probs, _ = model(
#                 t.states, forward=forward, goal_state_t=t.goal_state_t, mask=t.masks
#             )
#             nll = nll_loss(log_probs, t.actions, reduction="sum")
#             loss += nll
#             avg_action_nll += nll.item()
#             acc += (log_probs.detach().argmax(dim=1) == t.actions).sum().item()
#         else:
#             goal_feat = model.backward_feature_net(t.goal_state_t)
#             feats = model.backward_feature_net(t.states)
#             logits = model.backward_policy(feats, goal_feat)
#             if t.masks is not None:
#                 logits = logits.masked_fill(t.masks, -1e9)
#             log_probs = log_softmax(logits, dim=-1)

#             nll = nll_loss(log_probs, t.actions, reduction="sum")
#             loss += nll
#             avg_action_nll += nll.item()
#             acc += (log_probs.detach().argmax(dim=1) == t.actions).sum().item()

#             if n_subgoals > 0:
#                 k = min(n_subgoals, t_len - 1)
#                 subgoal_indices = to.randperm(t_len - 1)[
#                     :k
#                 ]  # need to add 1 to get the correct index
#                 for idx in subgoal_indices:
#                     idx += 1
#                     subg_actions = t.actions[:idx]
#                     logits = model.backward_policy(feats[:idx], feats[idx].unsqueeze(0))
#                     if t.masks is not None:
#                         logits = logits.masked_fill(t.masks[:idx], -1e9)
#                     log_probs = log_softmax(logits, dim=-1)

#                     nll = nll_loss(log_probs, subg_actions, reduction="sum")
#                     loss += nll
#                     avg_action_nll += nll.item()
#                     acc += (
#                         (log_probs.detach().argmax(dim=1) == subg_actions).sum().item()
#                     )
#                     total_actions += idx

#     loss /= total_actions
#     avg_action_nll /= total_actions
#     acc /= total_actions

#     return loss, avg_action_nll, acc


# # todo this is not correct, need to add masking?
# def merge_cross_entropy_loss(
#     trajs: list[Trajectory], model: AgentModel, n_subgoals: int = 0
# ):
#     forward = trajs[0].forward
#     merged_states = to.cat(tuple(t.states for t in trajs))
#     merged_actions = to.cat(tuple(t.actions for t in trajs))
#     merged_masks = to.cat(tuple(t.masks for t in trajs))

#     if forward:
#         merged_goal_states_t = None
#     else:
#         repeats = to.tensor(tuple(len(t) for t in trajs))
#         goal_states_t = to.cat(tuple(t.goal_state_t for t in trajs))
#         merged_goal_states_t = to.repeat_interleave(
#             goal_states_t, repeats=repeats, dim=0
#         )

#     log_probs, _ = model(
#         merged_states,
#         forward=forward,
#         goal_state_t=merged_goal_states_t,
#         mask=merged_masks,
#     )
#     loss = nll_loss(log_probs, merged_actions, reduction="mean")
#     avg_action_nll = loss.detach()
#     acc = (
#         (log_probs.detach().argmax(dim=1) == merged_actions)
#         .sum()
#         .div(len(log_probs))
#         .item()
#     )

#     return loss, avg_action_nll, acc
