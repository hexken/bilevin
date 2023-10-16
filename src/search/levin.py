from __future__ import annotations
import heapq
from timeit import default_timer as timer

import numpy as np
import torch as to

from domains.domain import State
from search.agent import Agent
from search.utils import Problem, SearchNode, Trajectory


class Levin(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def bidirectional(cls):
        return False

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

        problem_id = problem.id
        domain = problem.domain
        model = self.model
        cost_fn = self.cost_fn

        state = domain.reset()
        state_t = domain.state_tensor(state).unsqueeze(0)
        actions, mask = domain.actions_unpruned(state)
        log_probs, _ = model(state_t, mask=mask)

        node = SearchNode(
            state,
            parent=None,
            parent_action=None,
            actions=actions,
            actions_mask=mask,
            g_cost=0,
            log_prob=0.0,
            cost=0.0,
            log_action_probs=log_probs[0],
        )

        closed = {node: node}
        open = [node]

        num_expanded = 0
        while len(open) > 0:
            if (
                (exp_budget > 0 and num_expanded >= exp_budget)
                or time_budget > 0
                and timer() - start_time >= time_budget
            ):
                return num_expanded, 0, None

            node = heapq.heappop(open)
            num_expanded += 1

            masks = []
            children_to_be_evaluated = []
            state_t_of_children_to_be_evaluated = []
            for a in node.actions:
                new_state = domain.result(node.state, a)
                new_state_actions, mask = domain.actions(a, new_state)

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

                if new_node not in closed:
                    if domain.is_goal(new_state):
                        traj = Trajectory.from_goal_node(
                            domain=domain,
                            final_node=new_node,
                            num_expanded=num_expanded,
                            partial_g_cost=new_node.g_cost,
                            partial_log_prob=new_node.log_prob,
                            log_prob=new_node.log_prob,
                            model=model,
                        )
                        traj = (traj, None)
                        return num_expanded, 0, traj

                    closed[new_node] = new_node
                    if new_state_actions:
                        heapq.heappush(open, new_node)
                        children_to_be_evaluated.append(new_node)
                        state_t = domain.state_tensor(new_state)
                        state_t_of_children_to_be_evaluated.append(state_t)
                        masks.append(mask)

            if children_to_be_evaluated:
                children_state_t = to.stack(state_t_of_children_to_be_evaluated)
                masks = to.stack(masks)
                log_probs, _ = model(
                    children_state_t,
                    mask=masks,
                )

                for child, lap in zip(children_to_be_evaluated, log_probs):
                    child.log_action_probs = lap

        print(f"Emptied open list for problem {problem_id}")
        return num_expanded, 0, None
