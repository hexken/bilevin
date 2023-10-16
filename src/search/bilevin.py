from __future__ import annotations
import heapq
from timeit import default_timer as timer

import torch as to

from domains.domain import State
from enums import TwoDir
from search.agent import Agent
from search.utils import SearchNode, Problem


class BiLevin(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def bidirectional(cls):
        return True

    @property
    def trainable(cls):
        return True

    def search(
        self,
        problem: Problem,
        exp_budget: int,
        time_budget: float,
    ):
        """ """
        start_time = timer()

        model = self.model
        cost_fn = self.cost_fn

        problem_id = problem.id
        f_domain = problem.domain
        b_domain = f_domain.backward_domain()

        f_state = f_domain.reset()
        f_state_t = f_domain.state_tensor(f_state).unsqueeze(0)
        f_actions, f_mask = f_domain.actions_unpruned(f_state)
        f_log_probs, _ = model(f_state_t, mask=f_mask)
        f_start_node = SearchNode(
            f_state,
            parent=None,
            parent_action=None,
            actions=f_actions,
            actions_mask=f_mask,
            g_cost=0,
            log_prob=0.0,
            cost=0.0,
            log_action_probs=f_log_probs[0],
        )
        f_open = [f_start_node]
        f_closed = {f_start_node: f_start_node}
        f_domain.update(f_start_node)

        if model.requires_backward_goal:
            b_goal_feats = model.backward_feature_net(f_state_t)
        else:
            b_goal_feats = None

        b_state = b_domain.reset()
        b_state_t = b_domain.state_tensor(b_state).unsqueeze(0)
        b_actions, b_mask = b_domain.actions_unpruned(b_state)

        b_log_probs, _ = model(
            b_state_t, forward=False, goal_feats=b_goal_feats, mask=b_mask
        )
        b_start_node = SearchNode(
            b_state,
            parent=None,
            parent_action=None,
            actions=b_actions,
            actions_mask=b_mask,
            g_cost=0,
            log_prob=0.0,
            cost=0.0,
            log_action_probs=b_log_probs[0],
        )
        b_closed = {b_start_node: b_start_node}
        b_domain.update(b_start_node)
        b_open = [b_start_node]

        n_total_expanded = 0
        n_forw_expanded = 0
        n_backw_expanded = 0

        while len(f_open) > 0 and len(b_open) > 0:
            if (
                (exp_budget > 0 and n_total_expanded >= exp_budget)
                or time_budget > 0
                and timer() - start_time >= time_budget
            ):
                return (
                    n_forw_expanded,
                    n_backw_expanded,
                    None,
                )

            if f_open[0] < b_open[0]:
                direction = TwoDir.FORWARD
                _goal_feats = None
                _domain = f_domain
                _open = f_open
                _closed = f_closed
                other_domain = b_domain
            else:
                direction = TwoDir.BACKWARD
                _domain = b_domain
                _goal_feats = b_goal_feats
                _open = b_open
                _closed = b_closed
                other_domain = f_domain

            node = heapq.heappop(_open)
            if direction == TwoDir.FORWARD:
                n_forw_expanded += 1
            else:
                n_backw_expanded += 1
            n_total_expanded += 1

            masks = []
            children_to_be_evaluated = []
            state_t_of_children_to_be_evaluated = []
            for a in node.actions:
                new_state = _domain.result(node.state, a)
                new_state_actions, mask = _domain.actions(a, new_state)

                new_node = SearchNode(
                    new_state,
                    g_cost=node.g_cost + 1,
                    parent=node,
                    parent_action=a,
                    actions=new_state_actions,
                    actions_mask=mask,
                    log_prob=node.log_prob + node.log_action_probs[a].item(),
                )
                new_node.cost = cost_fn(new_node)

                if new_node not in _closed:
                    trajs = _domain.try_make_solution(
                        model,
                        new_node,
                        other_domain,
                        n_forw_expanded + n_backw_expanded,
                    )

                    if trajs:  # solution found
                        return (
                            n_forw_expanded,
                            n_backw_expanded,
                            trajs,
                        )

                    _closed[new_node] = new_node
                    _domain.update(new_node)

                    if new_state_actions:
                        heapq.heappush(_open, new_node)
                        children_to_be_evaluated.append(new_node)
                        state_t = _domain.state_tensor(new_state)
                        state_t_of_children_to_be_evaluated.append(state_t)
                        masks.append(mask)

            if children_to_be_evaluated:
                children_state_t = to.stack(state_t_of_children_to_be_evaluated)
                masks = to.stack(masks)
                log_probs, _ = model(
                    children_state_t,
                    forward=direction == TwoDir.FORWARD,
                    goal_feats=_goal_feats,
                    mask=masks,
                )

                for child, lap in zip(children_to_be_evaluated, log_probs):
                    child.log_action_probs = lap

        print(f"Emptied opens for problem {problem_id}")
        return (
            n_forw_expanded,
            n_backw_expanded,
            None,
        )
