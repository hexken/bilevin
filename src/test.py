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

import queue
import sys
import time
from typing import Optional, Union

import numpy as np
import pandas as pd
from tabulate import tabulate
import torch as to
import torch.distributed as dist
from torch.multiprocessing import Queue
from torch.utils.tensorboard.writer import SummaryWriter
import tqdm

from domains.domain import Problem
from search.agent import Agent
from loaders import ProblemsBatchLoader


def test(
    rank: int,
    agent: Agent,
    model: Optional[Union[to.nn.Module, tuple[to.nn.Module, to.nn.Module]]],
    problems_loader: ProblemsBatchLoader,
    writer: SummaryWriter,
    world_size: int,
    update_levin_costs: bool,
    initial_budget: int,
    results_queue: Queue,
    increase_budget: bool = True,
    print_batch_size: Optional[int] = None,
    print_results: bool = True,
    validate: bool = False,
    epoch: Optional[int] = None,
):
    if validate:
        print_results = False

    # print(f"rank {rank}")
    test_start_time = time.time()
    current_budget = initial_budget

    is_distributed = world_size > 1

    local_num_problems = len(problems_loader)
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

    dummy_data = np.column_stack(
        (
            np.zeros(
                (world_num_problems, len(search_result_header) - 1),
                dtype=np.int64,
            ),
        )
    )
    world_results_df = pd.DataFrame(dummy_data, columns=search_result_header[1:])
    del dummy_data
    world_results_df["Time"] = world_results_df["Time"].astype(float, copy=False)
    world_results_df["ProblemId"] = problems_loader.all_ids
    world_results_df.set_index("ProblemId", inplace=True)

    is_bidirectional = agent.bidirectional

    to.set_grad_enabled(False)
    if agent.trainable:
        if is_bidirectional:
            assert isinstance(model, tuple)
            f_model, b_model = model
            b_model.eval()  # type:ignore
        else:
            assert isinstance(model, to.nn.Module)
            f_model = model

        f_model.eval()

    total_num_expanded = 0

    world_solved_problems = set()
    local_remaining_problems = set(p[0] for p in problems_loader)

    def try_sync_results():
        """
        Try to sync results from the queue.
        Only rank 0 should call this function.
        *NOTE*: this mutates total_num_expanded and world_solved_problems
        """
        if print_batch_size:
            pbs = min(print_batch_size, world_num_problems - len(world_solved_problems))
            if results_queue.qsize() < pbs:
                return 0
            world_batch_results = [results_queue.get() for _ in range(pbs)]
        else:
            world_batch_results = []
            while True:
                try:
                    res = results_queue.get_nowait()
                    world_batch_results.append(res)
                    del res
                except queue.Empty:
                    break

        if len(world_batch_results) == 0:
            return 0

        world_batch_results_arr = np.vstack(world_batch_results)
        del world_batch_results
        world_batch_df = pd.DataFrame(
            {
                h: v
                for h, v in zip(
                    search_result_header[1:-3], world_batch_results_arr.T[1:-3]
                )
            }
        )
        world_batch_df["ProblemId"] = [
            problems_loader.all_ids[i] for i in world_batch_results_arr[:, 0]
        ]
        world_batch_df["StartTime"] = (
            (world_batch_results_arr[:, -3].astype(float) / 1000) - test_start_time
        ).round(3)
        world_batch_df["EndTime"] = (
            (world_batch_results_arr[:, -2].astype(float).round(3) / 1000)
            - test_start_time
        ).round(3)
        world_batch_df["Time"] = world_batch_results_arr[:, -1].astype(float) / 1000
        world_batch_df.set_index("ProblemId", inplace=True)
        world_batch_df.sort_values("NumExpanded", inplace=True)

        if print_results:
            print(
                tabulate(
                    world_batch_df,
                    headers="keys",
                    tablefmt="psql",
                )
            )

        batch_solved_df = world_batch_df[world_batch_df["SolutionLength"] > 0]
        for problem_id in batch_solved_df.index:
            assert problem_id not in world_solved_problems
            world_solved_problems.add(problem_id)

        if print_results:
            print(f"Solved {len(batch_solved_df)}/{len(world_batch_df)}\n")

        if validate:
            pbar.update(len(world_batch_df))
        else:
            pbar.update(len(batch_solved_df))

        sys.stdout.flush()

        nonlocal total_num_expanded
        total_num_expanded += world_batch_df["NumExpanded"].sum()

        if not validate:
            writer.add_scalar(
                f"cum_unique_solved_vs_expanded",
                len(world_solved_problems),
                total_num_expanded,
            )

        world_results_df.loc[batch_solved_df.index, :] = batch_solved_df

        return

    if rank == 0:
        pbar = tqdm.tqdm(total=world_num_problems)

    while True:
        # print(f"rank {rank} remaining: {len(local_remaining_problems)}")
        if rank == 0 and len(world_solved_problems) == world_num_problems:
            break
        elif rank != 0 and len(local_remaining_problems) == 0:
            break

        sync_toggle = False
        for problem in tuple(local_remaining_problems):
            # need to create a new results array, or else data could
            # be overwritten
            search_result = np.zeros(8, dtype=np.int64)
            # print(f"rank {rank} {problem.id}")
            start_time = time.time()
            (solution_length, num_expanded, num_generated, traj,) = agent.search(
                problem,
                model,
                current_budget,
                update_levin_costs,
            )
            end_time = time.time()
            if is_bidirectional:
                problem.domain.reset()

            start_time = int(start_time * 1000)
            end_time = int(end_time * 1000)

            search_result[0] = problem.id_idx
            search_result[1] = solution_length
            search_result[2] = current_budget
            search_result[3] = num_expanded
            search_result[4] = num_generated
            search_result[5] = start_time
            search_result[6] = end_time
            search_result[7] = end_time - start_time

            if traj:
                local_remaining_problems.remove(problem)

            results_queue.put(search_result)

            if rank == 0 and not validate:
                if sync_toggle:
                    try_sync_results()
                    if num_expanded > 0:
                        sync_toggle = False
                else:
                    sync_toggle = True

        if validate and is_distributed:
            dist.barrier()

        if rank == 0:
            try_sync_results()

        # epoch end
        if increase_budget:
            current_budget *= 2
        else:
            break

    if print_results:
        print(f"Rank {rank} finished at {time.ctime(time.time())}")

    if rank == 0:
        if epoch:
            fname = f"{writer.log_dir}/valid_{epoch}.csv"
        else:
            fname = f"{writer.log_dir}/results.csv"

        world_results_df.to_csv(fname)
        return len(world_solved_problems), total_num_expanded

    return None
