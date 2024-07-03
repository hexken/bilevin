from __future__ import annotations
from argparse import Namespace
from heapq import heappop
from pathlib import Path
from timeit import default_timer as timer
from typing import TYPE_CHECKING

import numpy as np
import torch as to

from enums import SearchDir
from search.agent import Agent
from search.loaders import Problem
from search.traj import from_goal_node

if TYPE_CHECKING:
    pass


class UniDir(Agent):
    def __init__(self, logdir: Path, args: Namespace, aux_args: dict):
        super().__init__(logdir, args, aux_args)
        self.n_eval = args.n_eval

    @property
    def is_bidirectional(self):
        return False

    def search(
        self,
        problem: Problem,
        exp_budget: int,
        time_budget: float,
    ):
        """ """
        start_time = timer()

        problem_id = problem.id
        domain = problem.domain

        state = domain.init()
        state_t = domain.state_tensor(state).unsqueeze(0)
        actions, mask = domain.actions_unpruned(state)
        node = self.make_start_node(
            state, state_t, actions, forward=True, mask=mask, goal_feats=None
        )

        closed = {node: node}
        open_list = [node]

        masks = []
        children_to_be_evaluated = []
        state_t_of_children_to_be_evaluated = []

        num_expanded = 0
        while len(open_list) > 0:
            if (
                (exp_budget > 0 and num_expanded >= exp_budget)
                or time_budget > 0
                and timer() - start_time >= time_budget
            ):
                return num_expanded, 0, (None, None)

            node = heappop(open_list)
            num_expanded += 1

            for a in node.actions:
                new_state = domain.result(node.state, a)
                new_state_actions, mask = domain.actions(a, new_state)

                new_node = self.make_partial_child_node(
                    node,
                    a,
                    new_state_actions,
                    mask,
                    new_state,
                )

                if new_node not in closed:
                    if domain.is_goal(new_state):
                        traj = from_goal_node(
                            self,
                            domain=domain,
                            goal_node=new_node,
                            num_expanded=num_expanded,
                            partial_g_cost=new_node.g,
                        )
                        traj = (traj, None)
                        return num_expanded, 0, traj

                    closed[new_node] = new_node
                    if new_state_actions:
                        children_to_be_evaluated.append(new_node)
                        state_t = domain.state_tensor(new_state)
                        state_t_of_children_to_be_evaluated.append(state_t)
                        masks.append(mask)

            if len(children_to_be_evaluated) >= self.n_eval or len(open_list) == 0:
                self.finalize_children_nodes(
                    open_list,
                    SearchDir.FORWARD,
                    children_to_be_evaluated,
                    state_t_of_children_to_be_evaluated,
                    masks,
                    None,
                )
                masks.clear()
                children_to_be_evaluated.clear()
                state_t_of_children_to_be_evaluated.clear()

        print(f"Emptied open list for problem {problem_id}")
        return num_expanded, 0, (None, None)
