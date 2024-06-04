from __future__ import annotations
from argparse import Namespace
from heapq import heappop, heappush
import heapq
from pathlib import Path
from timeit import default_timer as timer
from typing import TYPE_CHECKING

import torch as to

from enums import SearchDir
from search.agent import Agent
from search.node import DirStructures

if TYPE_CHECKING:
    from search.problem import Problem



class BiDirAlt(Agent):
    def __init__(self, logdir: Path, args: Namespace, aux_args: dict):
        super().__init__(logdir, args, aux_args)
        self.n_batch_expansions = args.n_batch_expansions

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
        f_state = f_domain.init()
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
        b_state = b_domain.init()
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

        f_dir_struct = DirStructures(
            SearchDir.FORWARD,
            f_open,
            f_closed,
            f_domain,
            b_domain,
            expanded=0,
        )
        f_start_node.dir_structures = f_dir_struct

        b_dir_struct = DirStructures(
            SearchDir.BACKWARD,
            b_open,
            b_closed,
            b_domain,
            f_domain,
            goal_feats=b_goal_feats,
            expanded=0,
        )
        b_start_node.dir_structures = b_dir_struct

        next_direction = SearchDir.FORWARD
        while len(f_open) > 0 and len(b_open) > 0:
            if (
                (exp_budget > 0 and n_total_expanded >= exp_budget)
                or time_budget > 0
                and timer() - start_time >= time_budget
            ):
                return (
                    f_dir_struct.expanded,
                    b_dir_struct.expanded,
                    None,
                )

            direction = next_direction

            if direction == SearchDir.FORWARD:
                next_direction = SearchDir.BACKWARD
                ds = f_dir_struct
            else:
                next_direction = SearchDir.FORWARD
                ds = b_dir_struct

            # try to expand b nodes
            nodes = []
            try:
                for _ in range(self.n_batch_expansions):
                    nodes.append(heappop(ds.open))
            except IndexError:
                pass

            ds.expanded += len(nodes)
            n_total_expanded += len(nodes)

            masks = []
            children_to_be_evaluated = []
            state_t_of_children_to_be_evaluated = []
            for node in nodes:
                for a in node.actions:
                    new_state = ds.domain.result(node.state, a)
                    new_state_actions, mask = ds.domain.actions(a, new_state)
                    new_node = self.make_partial_child_node(
                        node,
                        a,
                        new_state_actions,
                        mask,
                        new_state,
                    )

                    if new_node not in ds.closed:
                        trajs = ds.domain.try_make_solution(
                            self,
                            new_node,
                            ds.other_domain,
                            n_total_expanded,
                        )

                        if trajs is not None:  # solution found
                            return (
                                f_dir_struct.expanded,
                                b_dir_struct.expanded,
                                trajs,
                            )

                        ds.closed[new_node] = new_node
                        ds.domain.update(new_node)

                        if new_state_actions:
                            children_to_be_evaluated.append(new_node)
                            state_t = ds.domain.state_tensor(new_state)
                            state_t_of_children_to_be_evaluated.append(state_t)
                            masks.append(mask)

            if len(children_to_be_evaluated) > 0:
                self.finalize_children_nodes(
                    ds.open,
                    direction,
                    children_to_be_evaluated,
                    state_t_of_children_to_be_evaluated,
                    masks,
                    ds.goal_feats,
                )

        print(f"Emptied opens for problem {problem_id}")
        return (
            f_dir_struct.expanded,
            b_dir_struct.expanded,
            None,
        )
