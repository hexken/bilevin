from __future__ import annotations
from argparse import Namespace
from dataclasses import dataclass, field
from heapq import heappop, heappush, heapreplace
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
from sklearn.cluster import kmeans_plusplus
import torch as to
from torch.linalg import vector_norm as norm

from enums import SearchDir
from search.agent import Agent
from search.node import SearchNode
from search.problem import Problem


@dataclass(order=True)
class LandMark:
    dist: float
    state_feats: to.Tensor = field(compare=False)


class ApproxFF(Agent):
    def __init__(self, logdir: Path, args: Namespace, aux_args: dict):
        if args.loss_fn == "default":
            args.loss_fn = "metric_loss"
        self.n_landmarks = args.n_landmarks
        self.n_batch_expansions = args.n_batch_expansions
        super().__init__(logdir, args, aux_args)

    @property
    def is_bidirectional(self):
        return True

    @property
    def has_policy(self):
        return False

    @property
    def has_heuristic(self):
        return False

    def make_partial_child_node(self):
        pass

    def finalize_children_nodes(self):
        pass

    def make_start_node(self):
        pass

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
        f_actions, _ = f_domain.actions_unpruned(f_state)
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

        b_domain = f_domain.backward_domain()
        b_state = b_domain.init()
        b_actions, _ = b_domain.actions_unpruned(b_state)
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

        # forward OPEN init

        n_forw_expanded = 0
        curr_g = 0
        f_topen = [f_start_node]
        f_tclosed = {f_start_node}
        while f_topen[0].g == curr_g or len(f_topen) < self.n_batch_expansions:
            node = f_topen.pop(0)
            n_forw_expanded += 1
            for a in node.actions:
                new_state = f_domain.result(node.state, a)
                new_state_actions, _ = f_domain.actions(a, new_state)
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

                if new_node not in f_tclosed:
                    f_topen.append(new_node)
                    f_tclosed.add(new_node)

        open_state_ts = to.stack(
            [f_domain.state_tensor(node.state) for node in f_topen]
        )
        open_feats = self.model(open_state_ts)
        f_landmarks, _ = kmeans_plusplus(open_feats.numpy(), self.n_landmarks)
        f_landmarks = to.from_numpy(f_landmarks)

        # backward OPEN init
        n_backw_expanded = 0
        curr_g = 0
        b_topen = [b_start_node]
        b_tclosed = {b_start_node}
        while b_topen[0].g == curr_g or len(b_topen) < self.n_batch_expansions:
            node = b_topen.pop(0)
            n_backw_expanded += 1
            for a in node.actions:
                new_state = f_domain.result(node.state, a)
                new_state_actions, _ = f_domain.actions(a, new_state)
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

                if new_node not in b_tclosed:
                    b_topen.append(new_node)
                    b_tclosed.add(new_node)

        open_state_ts = to.stack(
            [b_domain.state_tensor(node.state) for node in b_topen]
        )
        open_feats = self.model(open_state_ts, forward=False)
        b_landmarks, _ = kmeans_plusplus(open_feats.numpy(), self.n_landmarks)
        b_landmarks = to.from_numpy(b_landmarks)

        f_open = [f_start_node]
        f_closed = {f_start_node: f_start_node}

        b_open = [b_start_node]
        b_closed = {b_start_node: b_start_node}

        f_landmark_heap = []
        for c in f_landmarks:
            dist = norm(b_landmarks - c, dim=1).min().item()
            heappush(f_landmark_heap, LandMark(dist, c))

        b_landmark_heap = []
        for c in b_landmarks:
            dist = norm(f_landmarks - c, dim=1).min().item()
            heappush(b_landmark_heap, LandMark(dist, c))

        n_total_expanded = n_forw_expanded + n_backw_expanded
        # print(f"Initial expanded: {n_total_expanded}")
        next_direction = SearchDir.FORWARD
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

            direction = next_direction

            if direction == SearchDir.FORWARD:
                next_direction = SearchDir.BACKWARD
                _landmarks = f_landmarks
                _landmark_heap = f_landmark_heap
                _other_landmarks = b_landmarks
                _domain = f_domain
                _open = f_open
                _closed = f_closed
                other_domain = b_domain
            else:
                next_direction = SearchDir.FORWARD
                _landmarks = b_landmarks
                _landmark_heap = b_landmark_heap
                _other_landmarks = f_landmarks
                _domain = b_domain
                _open = b_open
                _closed = b_closed
                other_domain = f_domain

            # try to expand b nodes
            nodes = []
            try:
                for _ in range(self.n_batch_expansions):
                    nodes.append(heappop(_open))
            except IndexError:
                pass

            if direction == SearchDir.FORWARD:
                n_forw_expanded += len(nodes)
            else:
                n_backw_expanded += len(nodes)
            n_total_expanded += len(nodes)

            children_to_be_evaluated = []
            state_t_of_children_to_be_evaluated = []
            for node in nodes:
                for a in node.actions:
                    new_state = _domain.result(node.state, a)
                    new_state_actions, _ = _domain.actions(a, new_state)
                    # partially constructed node
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

                    if new_node not in _closed:
                        trajs = _domain.try_make_solution(
                            self,
                            new_node,
                            other_domain,
                            n_total_expanded,
                            metric=True,
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

            # compute heuristics, update landmarks
            if len(children_to_be_evaluated) > 0:
                states_ts = to.stack(state_t_of_children_to_be_evaluated)
                states_feats = self.model(
                    states_ts, forward=direction == SearchDir.FORWARD
                )
                for i, child in enumerate(children_to_be_evaluated):
                    dist = norm(_other_landmarks - states_feats[i], dim=1).min().item()
                    child.f = dist
                    heappush(_open, child)
                    if dist <= _landmark_heap[0].dist:
                        heapreplace(_landmark_heap, LandMark(dist, states_feats[i]))

                new_landmarks = to.stack([c.state_feats for c in _landmark_heap])
                if direction == SearchDir.FORWARD:
                    f_landmarks = new_landmarks
                else:
                    b_landmarks = new_landmarks

        print(f"Emptied opens for problem {problem_id}")
        return (
            n_forw_expanded,
            n_backw_expanded,
            None,
        )
