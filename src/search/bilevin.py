import copy
import heapq
import math
import time

import numpy as np
import torch as to
import torch.nn.functional as F

from models.utils import mixture_uniform
from search.agent import Agent
from search.levin import LevinNode, levin_cost
from enums import TwoDir
from search.utils import get_merged_trajectory


class BiLevin(Agent):
    @property
    def bidirectional(cls):
        return True

    def __init__(
        self,
        weight_uniform=0.0,
    ):
        self.weight_uniform = weight_uniform

    def search(
        self,
        problem,
        problem_name,
        model,
        budget,
        learn=False,
        end_time=None,
    ):
        """ """
        device = next(model[0].parameters()).device
        b_problem = problem.backward_problem()

        f_state = problem.state_tensor(device)
        b_state = b_problem.state_tensor(device)
        initial_state = f_state.clone()
        dims = [1] * initial_state.dim()
        initial_state_repeated = initial_state.repeat(problem.num_actions, *dims)

        forward_model, backward_model = model

        f_action_logits = forward_model(f_state)
        b_action_logits = backward_model(b_state, initial_state)

        f_start_node = LevinNode(
            problem,
            g_cost=0,
            log_prob=1.0,
            levin_cost=1,
            log_action_probs=mixture_uniform(f_action_logits, self.weight_uniform),
            num_expanded_when_generated=0,
        )

        b_start_node = LevinNode(
            b_problem,
            g_cost=0,
            log_prob=1.0,
            levin_cost=1,
            log_action_probs=mixture_uniform(b_action_logits, self.weight_uniform),
            num_expanded_when_generated=0,
        )

        f_frontier = []
        b_frontier = []
        f_reached = {}
        b_reached = {}
        heapq.heappush(f_frontier, f_start_node)
        heapq.heappush(b_frontier, b_start_node)
        f_reached[f_start_node] = f_start_node
        b_reached[b_start_node] = b_start_node

        children_to_be_evaluated = []
        state_t_of_children_to_be_evaluated = []

        num_expanded = 0
        num_generated = 0
        while len(f_frontier) > 0 or len(b_frontier) > 0:
            # todo we don't consider problems where a queue could empty?

            if (
                (budget and num_expanded >= budget)
                or end_time
                and time.time() > end_time
            ):
                return (False, num_expanded, num_generated, None)

            if b_frontier[0] < f_frontier[0]:
                direction = TwoDir.BACKWARD
                _model = backward_model
                _frontier = b_frontier
                _reached = b_reached
                _other_reached = f_reached
            else:
                direction = TwoDir.FORWARD
                _model = forward_model
                _frontier = f_frontier
                _reached = f_reached
                _other_reached = b_reached

            node = heapq.heappop(_frontier)
            num_expanded += 1
            actions = node.state.successors_parent_pruning(node.action)
            for a in actions:
                # todo vectorize this? Will depend on how I re-implement envs
                new_state = copy.deepcopy(node.state)
                new_state.apply_action(a)

                new_node = LevinNode(
                    new_state,
                    node,
                    a,
                    node.g_cost + 1,
                    node.log_prob + node.log_action_probs[a],
                    num_expanded_when_generated=num_expanded,
                )
                num_generated += 1

                if new_node not in _reached:
                    _reached[new_node] = new_node
                    # todo
                    if new_node in _other_reached:  # solution found
                        f_common_node = f_reached[new_node]
                        b_common_node = b_reached[new_node]

                        forward_traj = get_merged_trajectory(
                            f_common_node,
                            b_common_node,
                            LevinNode,
                            num_expanded,
                        )

                        if learn:
                            backward_traj = get_merged_trajectory(
                                b_common_node,
                                f_common_node,
                                LevinNode,
                                num_expanded,
                            )
                            trajs = forward_traj, backward_traj
                        else:
                            trajs = forward_traj

                        return len(forward_traj), num_expanded, num_generated, trajs

                children_to_be_evaluated.append(new_node)
                state_t_of_children_to_be_evaluated.append(
                    new_state.state_tensor(device)
                )

            batch_states = to.stack(state_t_of_children_to_be_evaluated).to(device)
            if direction == TwoDir.BACKWARD:
                action_logits = _model(
                    batch_states, initial_state_repeated[: len(batch_states)]
                )
            else:
                action_logits = _model(batch_states)

            log_action_probs = mixture_uniform(action_logits, self.weight_uniform)

            for i, new_node in enumerate(children_to_be_evaluated):
                # todo vectorize?

                lc = levin_cost(new_node)
                new_node.log_action_probs = log_action_probs[i]
                new_node.levin_cost = lc  # type:ignore
                heapq.heappush(_frontier, new_node)

            children_to_be_evaluated = []
            state_t_of_children_to_be_evaluated = []

        print("Emptied Open List in problem: ", problem_name)
        return False, num_expanded, num_generated, None
