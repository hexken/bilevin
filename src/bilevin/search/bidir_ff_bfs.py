from __future__ import annotations
import heapq
from timeit import default_timer as timer

import torch as to

from enums import SearchDir
from search.agent import Agent
from search.node import SearchNode
from search.problem import Problem


class BiDirFFBFS(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = 4
        self.b = 16

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
        n_total_expanded = 0

        problem_id = problem.id
        f_domain = problem.domain
        f_state = f_domain.init()
        f_actions, f_mask = f_domain.actions_unpruned(f_state)
        f_start_node = SearchNode(
            f_state,
            parent=None,
            parent_action=None,
            actions=f_actions,
            actions_mask=None,
            g=0,
            f=0.0,
            log_prob=0.0,
        )

        # forward frontier init
        f_open = [f_start_node]
        f_closed = {f_start_node: f_start_node}
        queue = [f_start_node]
        n_forw_expanded = 0
        while len(f_open) < self.b:
            node = queue.pop(0)
            n_forw_expanded += 1
            for a in node.actions:
                new_state = f_domain.result(node.state, a)
                new_state_actions, mask = f_domain.actions(a, new_state)
                new_node = SearchNode(
                    new_state,
                    parent=node,
                    parent_action=a,
                    actions=new_state_actions,
                    actions_mask=None,
                    g=node.g + 1,
                    f=0.0,
                    log_prob=0.0,
                )
                queue.append(new_node)
                f_closed[new_node] = new_node
                f_domain.update(new_node)
                f_open.append(new_node)

        b_domain = f_domain.backward_domain()
        b_state = b_domain.init()
        b_actions, b_mask = b_domain.actions_unpruned(b_state)

        b_start_node = SearchNode(
            b_state,
            parent=None,
            parent_action=None,
            actions=b_actions,
            actions_mask=None,
            g=0,
            f=0.0,
            log_prob=0.0,
        )

        b_open = [b_start_node]
        b_closed = {b_start_node: b_start_node}
        queue = [b_start_node]
        n_backw_expanded = 0
        while len(b_open) < self.b:
            node = queue.pop(0)
            n_backw_expanded += 1
            for a in node.actions:
                new_state = b_domain.result(node.state, a)
                new_state_actions, mask = b_domain.actions(a, new_state)
                new_node = SearchNode(
                    new_state,
                    parent=node,
                    parent_action=a,
                    actions=new_state_actions,
                    actions_mask=None,
                    g=node.g + 1,
                    f=0.0,
                    log_prob=0.0,
                )
                queue.append(new_node)
                b_closed[new_node] = new_node
                b_domain.update(new_node)
                b_open.append(new_node)

        # compute centers

        # sort open lists
        n_total_expanded = n_forw_expanded + n_backw_expanded
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

            if b_open[0] < f_open[0]:
                direction = SearchDir.BACKWARD
                _domain = b_domain
                _goal_feats = b_goal_feats
                _open = b_open
                _closed = b_closed
                other_domain = f_domain
                n_backw_expanded += 1
            else:
                direction = SearchDir.FORWARD
                _goal_feats = None
                _domain = f_domain
                _open = f_open
                _closed = f_closed
                other_domain = b_domain
                n_forw_expanded += 1

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
