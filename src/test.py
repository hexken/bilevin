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

import time
from typing import Optional, Union

import numpy as np
import pandas as pd
from tabulate import tabulate
import torch as to
import torch.distributed as dist
from torch.utils.tensorboard.writer import SummaryWriter
import tqdm

from domains.domain import Problem
from loaders import ProblemsBatchLoader
from search import Agent
from models import AgentModel


def test(
    rank: int,
    agent: Agent,
    model: AgentModel,
    problems_loader: ProblemsBatchLoader,
    writer: SummaryWriter,
    world_size: int,
    update_levin_costs: bool,
    initial_budget: int,
    increase_budget: bool = True,
    print_results: bool = True,
    validate: bool = False,
    epoch: Optional[int] = None,
):
    if not epoch:
        epoch = 1

    test_start_time = time.time()
    current_budget = initial_budget

    is_distributed = world_size > 1

    world_num_problems = len(problems_loader.all_ids)

    search_result_header = [
        "ProblemId",
        "SolutionLength",
        "Budget",
        "NumExpanded",
        "NumGenerated",
        "StartTime",
        "EndTime",
        "Time",
    ]

    is_bidirectional = agent.bidirectional

    to.set_grad_enabled(False)
    if agent.trainable:
        model.eval()

    total_num_expanded = 0

    world_solved_problems = set()
    local_remaining_problems = set(p[0] for p in problems_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=world_num_problems)

    while True:
        # print(f"rank {rank} remaining: {len(local_remaining_problems)}")
        if rank == 0 and len(world_solved_problems) == world_num_problems:
            break
        elif rank != 0 and len(local_remaining_problems) == 0:
            break

        local_search_results = to.zeros(
            (len(local_remaining_problems), len(search_result_header)), dtype=to.int64
        )

        num_problems_t = to.zeros(world_size, dtype=to.int64)
        num_problems_t[rank] = len(local_remaining_problems)
        dist.reduce(num_problems_t, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            world_search_results = [
                to.zeros(i.item(), len(search_result_header), dtype=to.int64)
                for i in num_problems_t
            ]
        else:
            world_search_results = None

        for i, problem in enumerate(tuple(local_remaining_problems)):
            # need to create a new results array, or else data could
            # be overwritten
            # print(f"rank {rank} {problem.id}")
            start_time = int(time.time() * 1000)
            (solution_length, num_expanded, num_generated, traj,) = agent.search(
                problem,
                model,
                current_budget,
                update_levin_costs,
            )
            end_time = int(time.time() * 1000)

            if is_bidirectional:
                problem.domain.reset()

            local_search_results[i, 0] = problem.id_idx
            local_search_results[i, 1] = solution_length
            local_search_results[i, 2] = current_budget
            local_search_results[i, 3] = num_expanded
            local_search_results[i, 4] = num_generated
            local_search_results[i, 5] = start_time
            local_search_results[i, 6] = end_time
            local_search_results[i, 7] = end_time - start_time

            if traj:
                local_remaining_problems.remove(problem)

        if is_distributed:
            dist.barrier()

        dist.gather(local_search_results, world_search_results)

        if rank == 0:
            world_search_results_arr = to.vstack(world_search_results).numpy()

            world_results_df = pd.DataFrame(
                world_search_results_arr[:, 1:-3], columns=search_result_header[1:-3]
            )

            world_results_df["ProblemId"] = [
                problems_loader.all_ids[i] for i in world_search_results_arr[:, 0]
            ]
            world_results_df.set_index("ProblemId", inplace=True)

            world_results_df["StartTime"] = (
                (world_search_results_arr[:, -3].astype(float) / 1000) - test_start_time
            ).round(3)
            world_results_df["EndTime"] = (
                (world_search_results_arr[:, -2].astype(float).round(3) / 1000)
                - test_start_time
            ).round(3)
            world_results_df["Time"] = (
                world_search_results_arr[:, -1].astype(float) / 1000
            )

            solved_ids = world_results_df[world_results_df["SolutionLength"] > 0].index
            for problem_id in solved_ids:
                assert problem_id not in world_solved_problems
                world_solved_problems.add(problem_id)

            if print_results:
                print(
                    tabulate(
                        world_results_df,
                        headers="keys",
                        tablefmt="psql",
                    )
                )
                print(f"Solved {len(solved_ids)}/{world_num_problems}\n")

            world_results_df.sort_values("NumExpanded", inplace=True)
            total_num_expanded += world_results_df["NumExpanded"].sum()

            if not validate:
                writer.add_scalar(
                    f"cum_unique_solved_vs_expanded",
                    len(world_solved_problems),
                    total_num_expanded,
                )

            if validate:
                pbar.update(world_num_problems)
                fname = f"{writer.log_dir}/valid_{epoch}.csv"
            else:
                pbar.update(len(solved_ids))
                fname = f"{writer.log_dir}/test_{epoch}.csv"

            world_results_df.to_csv(fname)

        epoch += 1
        if increase_budget:
            current_budget *= 2
        else:
            break

    if rank == 0:
        return len(world_solved_problems), total_num_expanded

    return None
