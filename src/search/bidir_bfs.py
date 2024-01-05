from __future__ import annotations
import heapq
from timeit import default_timer as timer

import torch as to

from enums import TwoDir
from search.agent import Agent
from search.utils import Problem


class BiDirBFS(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def is_bidirectional(self):
        return True

    def search(
        self,
        problem: Problem,
        exp_budget: int,
        time_budget: float,
    ):
        """ """
        start_time = timer()

        problem_id = problem.id
        f_domain = problem.domain
        f_state = f_domain.reset()
        f_state_t = f_domain.state_tensor(f_state).unsqueeze(0)
        f_actions, f_mask = f_domain.actions_unpruned(f_state)
        f_start_node = self.make_start_node(
            f_state, f_state_t, f_actions, mask=f_mask, forward=True, goal_feats=None
        )
        f_open = [f_start_node]
        f_closed = {f_start_node: f_start_node}
        f_domain.update(f_start_node)

        if self.model.conditional_backward:
            if self.model.has_feature_net:
                b_goal_feats = self.model.backward_feature_net(f_state_t)
            else:
                b_goal_feats = f_state_t.flatten()
        else:
            b_goal_feats = None

        b_domain = f_domain.backward_domain()
        b_state = b_domain.reset()
        b_state_t = b_domain.state_tensor(b_state).unsqueeze(0)
        b_actions, b_mask = b_domain.actions_unpruned(b_state)

        b_start_node = self.make_start_node(
            b_state,
            b_state_t,
            b_actions,
            mask=b_mask,
            forward=False,
            goal_feats=b_goal_feats,
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
                n_forw_expanded += 1
            else:
                direction = TwoDir.BACKWARD
                _domain = b_domain
                _goal_feats = b_goal_feats
                _open = b_open
                _closed = b_closed
                other_domain = f_domain
                n_backw_expanded += 1

            node = heapq.heappop(_open)
            n_total_expanded += 1

            masks = []
            children_to_be_evaluated = []
            state_t_of_children_to_be_evaluated = []
            for a in node.actions:
                new_state = _domain.result(node.state, a)
                new_state_actions, mask = _domain.actions(a, new_state)
                new_node = self.make_partial_child_node(
                    node,
                    a,
                    new_state_actions,
                    mask,
                    new_state,
                )

                if new_node not in _closed:
                    trajs = _domain.try_make_solution(
                        self,
                        new_node,
                        other_domain,
                        n_total_expanded,
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
                        children_to_be_evaluated.append(new_node)
                        state_t = _domain.state_tensor(new_state)
                        state_t_of_children_to_be_evaluated.append(state_t)
                        masks.append(mask)

            if len(children_to_be_evaluated) > 0:
                self.finalize_children_nodes(
                    _open,
                    direction,
                    children_to_be_evaluated,
                    state_t_of_children_to_be_evaluated,
                    masks,
                    _goal_feats,
                )

        print(f"Emptied opens for problem {problem_id}")
        return (
            n_forw_expanded,
            n_backw_expanded,
            None,
        )
