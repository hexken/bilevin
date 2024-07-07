from __future__ import annotations
from argparse import Namespace
from heapq import heappop, heappush
from pathlib import Path
from timeit import default_timer as timer
from typing import Optional, TYPE_CHECKING

import torch as to

from enums import SearchDir
from search.agent import Agent
from search.loaders import Problem
from search.node import DirStructures

if TYPE_CHECKING:
    pass


class BiDir(Agent):
    def __init__(
        self, logdir: Path, args: Namespace, aux_args: dict, alternating: bool = True
    ):
        super().__init__(logdir, args, aux_args)
        self.n_eval = args.n_eval
        self.alternating = alternating

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
        # print(f"alternating: {self.alternating}")

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

        num_expanded = 0

        f_ds = DirStructures(SearchDir.FORWARD, f_open, f_closed, f_domain, b_domain)
        f_start_node.ds = ds = f_ds

        b_ds = DirStructures(
            SearchDir.BACKWARD,
            b_open,
            b_closed,
            b_domain,
            f_domain,
            goal_feats=b_goal_feats,
        )
        b_start_node.ds = ds = b_ds

        f_ds.next_ds = b_ds
        b_ds.next_ds = f_ds
        ds = f_ds

        while len(f_open) > 0 and len(b_open) > 0:
            if (
                (exp_budget > 0 and num_expanded >= exp_budget)
                or time_budget > 0
                and timer() - start_time >= time_budget
            ):
                return (
                    f_ds.expanded,
                    b_ds.expanded,
                    (None, None),
                )

            # flen = len(f_open)
            # blen = len(b_open)
            # if flen == 0 and blen > 0:
            #     node = heappop(b_open)
            # elif blen == 0 and flen > 0:
            #     node = heappop(f_open)
            # elif flen == 0 and blen == 0:
            #     break
            # else:

            if self.alternating:
                # if len(ds.open) == 0:
                #     ds = ds.next_ds
                node = heappop(ds.open)
            else:
                flen = len(f_open)
                blen = len(b_open)
                if flen == 0:
                    node = heappop(b_open)
                elif blen == 0:
                    node = heappop(f_open)
                elif b_open[0] < f_open[0]:
                    node = heappop(b_open)
                else:
                    node = heappop(f_open)

            num_expanded += 1
            ds.expanded += 1

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
                new_node.ds = ds

                if new_node not in ds.closed:
                    f_traj, b_traj = ds.domain.try_make_solution(
                        self,
                        new_node,
                        ds.other_domain,
                        num_expanded,
                    )

                    if f_traj is not None:  # solution found
                        return (
                            f_ds.expanded,
                            b_ds.expanded,
                            (f_traj, b_traj),
                        )

                    ds.closed[new_node] = new_node
                    ds.domain.update(new_node)

                    if new_state_actions:
                        state_t = ds.domain.state_tensor(new_state)
                        ds.children_to_be_evaluated.append(new_node)
                        ds.state_t_of_children_to_be_evaluated.append(state_t)
                        ds.masks.append(mask)

            if (
                len(f_ds.children_to_be_evaluated) + len(b_ds.children_to_be_evaluated)
                >= self.n_eval
                or len(ds.next_ds.open) == 0
            ):
                for _ds in (f_ds, b_ds):
                    if len(_ds.children_to_be_evaluated) > 0:
                        self.finalize_children_nodes(
                            _ds.open,
                            _ds.dir,
                            _ds.children_to_be_evaluated,
                            _ds.state_t_of_children_to_be_evaluated,
                            _ds.masks,
                            _ds.goal_feats,
                        )
                        _ds.children_to_be_evaluated.clear()
                        _ds.state_t_of_children_to_be_evaluated.clear()
                        _ds.masks.clear()
                ds = ds.next_ds

        assert False
        print(f"Emptied opens for problem {problem.id}")
        return (
            f_ds.expanded,
            b_ds.expanded,
            (None, None),
        )
