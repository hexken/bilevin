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

from timeit import default_timer as timer
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from tabulate import tabulate
import torch as to
import torch.distributed as dist
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from loaders import ProblemsBatchLoader
from search import Agent
from search.utils import int_columns, search_result_header


def test(
    rank: int,
    agent: Agent,
    problems_loader: ProblemsBatchLoader,
    writer: SummaryWriter,
    world_size: int,
    expansion_budget: int,
    time_budget: int,
    increase_budget: bool = True,
    print_results: bool = True,
    validate: bool = False,
    epoch: Optional[int] = None,
):
    if not epoch:
        epoch = 1

    current_budget = expansion_budget

    is_distributed = world_size > 1

    world_num_problems = len(problems_loader.all_ids)

    bidirectional = agent.bidirectional
    model = agent.model

    to.set_grad_enabled(False)
    if agent.trainable:
        model.eval()

    total_num_expanded = 0

    world_solved_problems = set()
    local_remaining_problems = set()
    local_remaining_problems = set(p[0] for p in problems_loader if p)

    fb_exp_ratio = -1
    fb_g_ratio = -1

    if rank == 0:
        print("Testing...")
    test_start_time = timer()
    while True:
        num_solved_t = to.zeros(1, dtype=to.int64)
        num_solved_t[0] = len(world_solved_problems)
        if is_distributed:
            dist.broadcast(num_solved_t, src=0)
        if num_solved_t.item() == world_num_problems:
            break

        local_search_results = np.zeros(
            (len(local_remaining_problems), len(search_result_header)), dtype=np.float64
        )

        if rank == 0:
            world_search_results = [None] * world_size
        else:
            world_search_results = None

        for i, problem in enumerate(tuple(local_remaining_problems)):
            start_time = timer()
            (
                n_forw_expanded,
                n_backw_expanded,
                n_forw_generated,
                n_backw_generated,
                traj,
            ) = agent.search(
                problem,
                current_budget,
                time_budget=time_budget,
            )
            end_time = timer()
            solution_length = 0 if not traj else traj[0].cost

            if bidirectional:
                problem.domain.reset()

            local_search_results[i, 0] = problem.id_idx
            local_search_results[i, 1] = end_time - start_time
            local_search_results[i, 2] = n_forw_expanded + n_backw_expanded
            local_search_results[i, 3] = n_forw_expanded
            local_search_results[i, 4] = n_backw_expanded
            local_search_results[i, 5] = n_forw_generated + n_backw_generated
            local_search_results[i, 6] = n_forw_generated
            local_search_results[i, 7] = n_backw_generated
            local_search_results[i, 8] = solution_length

            if traj:
                local_search_results[i, 9] = traj[0].partial_g_cost
                local_search_results[i, 11] = -1 * traj[0].partial_log_prob
                local_search_results[i, 13] = -1 * traj[0].log_prob
                if bidirectional:
                    local_search_results[i, 10] = traj[1].partial_g_cost
                    local_search_results[i, 12] = -1 * traj[1].partial_log_prob
                    local_search_results[i, 14] = -1 * traj[1].log_prob
            local_search_results[i, 15] = end_time - test_start_time

        if is_distributed:
            dist.barrier()

        if is_distributed:
            dist.gather_object(local_search_results, world_search_results)
        else:
            world_search_results = [local_search_results]

        if rank == 0:
            world_search_results_arr = np.vstack(world_search_results)

            world_results_df = pd.DataFrame(
                world_search_results_arr[:, 1:], columns=search_result_header[1:]
            )
            for col in int_columns:
                world_results_df[col] = world_results_df[col].astype(int)

            world_results_df["ProblemId"] = [
                problems_loader.all_ids[i]
                for i in world_search_results_arr[:, 0].astype(int)
            ]
            world_results_df = world_results_df.set_index("ProblemId")

            solved_ids = world_results_df[world_results_df["Len"] > 0].index
            for problem_id in solved_ids:
                assert problem_id not in world_solved_problems
                world_solved_problems.add(problem_id)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                fb_exp_ratio = (
                    world_results_df["FExp"].sum() / world_results_df["BExp"].sum()
                )
                fb_g_ratio = world_results_df["Fg"].sum() / world_results_df["Bg"].sum()

            world_results_df.sort_values("Exp")
            if print_results:
                print(
                    tabulate(
                        world_results_df,
                        headers="keys",
                        tablefmt="psql",
                    )
                )
                print(f"{'Solved':23s}: {len(solved_ids)}/{world_num_problems}\n")
                print(f"{'F/B expansion ratio':23s}: {fb_exp_ratio:.3f}")
                print(f"{'F/B g-cost ratio':23s}: {fb_g_ratio:.3f}\n")

            total_num_expanded += world_results_df["Exp"].sum()

            print(
                f"Solved: {len(solved_ids)}/{world_num_problems} in {timer() - test_start_time:.2f}s"
            )
            if validate:
                fname = f"{writer.log_dir}/epoch-{epoch}/valid.pkl"
            else:
                writer.add_scalar(
                    f"cum_unique_solved_vs_expanded",
                    len(world_solved_problems),
                    total_num_expanded,
                )
                fname = f"{writer.log_dir}/test.pkl"

            world_results_df.to_pickle(fname)

        epoch += 1
        if increase_budget:
            current_budget *= 2
            if rank == 0:
                print(
                    f"Budget increased from {current_budget / 2} to {current_budget} "
                )
        else:
            break

    if rank == 0:
        return len(world_solved_problems), total_num_expanded, fb_exp_ratio, fb_g_ratio

    return None, None, None, None
