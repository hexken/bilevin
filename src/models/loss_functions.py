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
    total_states = 0
    avg_action_nll = 0
    for t in trajs:
        log_probs, _ = model(
            t.states, forward=t.forward, goal_state_t=t.goal_state_t, mask=t.masks
        )
        nll = nll_loss(log_probs, t.actions, reduction="sum")
        loss += nll * t.num_expanded

        avg_action_nll += nll.item()
        acc += (log_probs.detach().argmax(dim=1) == t.actions).sum().item()
        total_states += len(t)

    loss /= len(trajs)
    avg_action_nll /= total_states
    acc /= total_states

    return loss, avg_action_nll, acc


# def levin_loss_scatter(trajs: MergedTrajectory, model: AgentModel):
#     state_feats = model.feature_net(trajs.states)

#     if trajs.forward:
#         logits = model.forward_policy(state_feats)
#     else:
#         if trajs.goal_states is not None:
#             goal_feats = model.feature_net(trajs.goal_states)
#             goal_feats_expanded = to.repeat_interleave(goal_feats, trajs.lengths, dim=0)
#             assert goal_feats_expanded.shape[0] == state_feats.shape[0]
#             logits = model.backward_policy(state_feats, goal_feats_expanded)
#         else:
#             logits = model.backward_policy(state_feats)

#     action_nlls = F.cross_entropy(logits, trajs.actions, reduction="none")
#     traj_nlls = ts.scatter_add(action_nlls, trajs.indices, dim=0)
#     loss = to.dot(traj_nlls, trajs.nums_expanded) / trajs.num_trajs
#     avg_action_nll = action_nlls.detach().mean().item()

#     return loss, avg_action_nll, logits.detach()


# def trajs_nlls(trajs: MergedTrajectory, model: AgentModel):
#     state_feats = model.feature_net(trajs.states)

#     if trajs.forward:
#         logits = model.forward_policy(state_feats)
#     else:
#         if trajs.goal_states is not None:
#             goal_feats = model.feature_net(trajs.goal_states)
#             goal_feats_expanded = to.repeat_interleave(goal_feats, trajs.lengths, dim=0)
#             assert goal_feats_expanded.shape[0] == state_feats.shape[0]
#             logits = model.backward_policy(state_feats, goal_feats_expanded)
#         else:
#             logits = model.backward_policy(state_feats)

#     action_nlls = F.cross_entropy(logits, trajs.actions, reduction="none")

#     traj_nlls = ts.scatter_add(action_nlls, trajs.indices, dim=0)
#     start_idx = 0
#     action_nlls_d = action_nlls.detach()
#     traj_partial_nlls = to.zeros(trajs.num_trajs)
#     for i in range(trajs.num_trajs):
#         partial_action_nlls = action_nlls_d[start_idx : trajs.steps[i]]
#         partial_nll = partial_action_nlls.sum()
#         traj_partial_nlls[i] = partial_nll
#         start_idx += trajs.lengths[i] - 1

#     return traj_nlls.detach(), traj_partial_nlls


# def ub_loss(trajs: MergedTrajectory, model: AgentModel):
#     pass
# state_feats = model.feature_net(trajs.states)

# if trajs.forward:
#     logits = model.forward_policy(state_feats)
# else:
#     if trajs.goal_states is not None:
#         goal_feats = model.feature_net(trajs.goal_states)
#         goal_feats_expanded = to.repeat_interleave(goal_feats, trajs.lengths, dim=0)
#         assert goal_feats_expanded.shape[0] == state_feats.shape[0]
#         logits = model.backward_policy(state_feats, goal_feats_expanded)
#     else:
#         logits = model.backward_policy(state_feats)

# action_nlls = F.cross_entropy(logits, trajs.actions, reduction="none")
# traj_nlls = ts.scatter_add(action_nlls, trajs.indices, dim=0)
# probs = to.exp(-1 * traj_nlls.detach())
# upperbounds = to.div(trajs.lengths + 1, probs)
# loss = to.dot(traj_nlls, upperbounds) / trajs.num_trajs
# avg_action_nll = action_nlls.detach().mean().item()

# return loss, avg_action_nll, logits.detach()


# def min_num_actions_ub_loss(trajs: MergedTrajectory, model: AgentModel):
#     pass
# state_feats = model.feature_net(trajs.states)

# if trajs.forward:
#     logits = model.forward_policy(state_feats)
# else:
#     if trajs.goal_states is not None:
#         goal_feats = model.feature_net(trajs.goal_states)
#         goal_feats_expanded = to.repeat_interleave(goal_feats, trajs.lengths, dim=0)
#         assert goal_feats_expanded.shape[0] == state_feats.shape[0]
#         logits = model.backward_policy(state_feats, goal_feats_expanded)
#     else:
#         logits = model.backward_policy(state_feats)

# action_nlls = F.cross_entropy(logits, trajs.actions, reduction="none")
# traj_nlls = ts.scatter_add(action_nlls, trajs.indices, dim=0)
# upperbounds = to.div(trajs.lengths, trajs.probs)
# loss = to.dot(traj_nlls, upperbounds) / trajs.num_trajs
# avg_action_nll = action_nlls.detach().mean().item()

# return loss, avg_action_nll, logits.detach()


# def cross_entropy_loss(trajs: MergedTrajectory, model: AgentModel):
#     state_feats = model.feature_net(trajs.states)

#     if trajs.forward:
#         logits = model.forward_policy(state_feats)
#     else:
#         if trajs.goal_states is not None:
#             goal_feats = model.feature_net(trajs.goal_states)
#             goal_feats_expanded = to.repeat_interleave(goal_feats, trajs.lengths, dim=0)
#             assert goal_feats_expanded.shape[0] == state_feats.shape[0]
#             logits = model.backward_policy(state_feats, goal_feats_expanded)
#         else:
#             logits = model.backward_policy(state_feats)

#     action_nlls = F.cross_entropy(logits, trajs.actions, reduction="none")
#     loss = action_nlls.sum() / trajs.num_trajs
#     avg_action_nll = action_nlls.detach().mean().item()

#     return loss, avg_action_nll, logits.detach()
