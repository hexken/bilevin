import time
from typing import Union

import numpy as np
import pandas as pd
from tabulate import tabulate
import torch as to
import torch.distributed as dist
from torch.multiprocessing import Queue
from torch.utils.tensorboard.writer import SummaryWriter
import tqdm

from domains.domain import Problem


def test(
    agent,
    model: Union[to.nn.Module, tuple[to.nn.Module, to.nn.Module]],
    problems: list[Problem],
    print_batch_size: int,
    writer: SummaryWriter,
    world_size: int,
    update_levin_costs: bool,
    initial_budget: int,
    results_queue: Queue,
):
    test_start_time = time.time()
    current_budget = initial_budget

    is_distributed = world_size > 1
    local_num_problems = len(problems)

    if is_distributed:

        rank = dist.get_rank()

        sh_t = to.zeros(1, dtype=to.int32) + local_num_problems
        dist.all_reduce(sh_t, dist.ReduceOp.SUM)
        world_num_problems = int(sh_t[0].item())
    else:
        world_num_problems = local_num_problems
        rank = 0

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
            range(world_num_problems),
            np.zeros(
                (world_num_problems, len(search_result_header) - 1), dtype=np.int32
            ),
        )
    )
    world_results_df = pd.DataFrame(dummy_data, columns=search_result_header)
    del dummy_data
    world_results_df["Time"] = world_results_df["Time"].astype(float, copy=False)
    world_results_df["StartTime"] = world_results_df["StartTime"].astype(
        float, copy=False
    )
    world_results_df["EndTime"] = world_results_df["EndTime"].astype(float, copy=False)
    world_results_df.set_index("ProblemId", inplace=True)

    is_bidirectional = agent.bidirectional
    if is_bidirectional:
        assert isinstance(model, tuple)
        f_model, b_model = model
    else:
        assert isinstance(model, to.nn.Module)
        f_model = model

    to.set_grad_enabled(False)
    f_model.eval()
    if is_bidirectional:
        b_model.eval()  # type:ignore

    total_num_expanded = 0

    world_solved_problems = set()
    local_solved_problems = set()

    def try_sync_results():
        """
        Try to sync results from the queue.
        Only rank 0 should call this function.
        """
        pbs = min(print_batch_size, world_num_problems - len(world_solved_problems))
        qsize = results_queue.qsize()
        if qsize < pbs:
            return 0

        world_batch_results = [results_queue.get() for _ in range(pbs)]
        world_batch_results_arr = np.vstack(world_batch_results)
        world_batch_df = pd.DataFrame(
            {
                h: v
                for h, v in zip(
                    search_result_header[0:-3], world_batch_results_arr.T[:-3]
                )
            }
        )
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

        print(f"Solved {len(batch_solved_df)}/{len(world_batch_results)}\n")

        pbar.update(len(batch_solved_df))
        writer.add_scalar(
            f"cum_unique_solved_vs_time",
            len(world_solved_problems),
            time.time(),
        )
        writer.add_scalar(
            f"cum_unique_solved_vs_expanded",
            len(world_solved_problems),
            total_num_expanded,
        )

        world_results_df.loc[batch_solved_df.index, :] = batch_solved_df

        return world_batch_df["NumExpanded"].sum()

    if rank == 0:
        pbar = tqdm.tqdm(total=world_num_problems)

    while True:
        local_remaining_problems = tuple(
            p for p in problems if p.id not in local_solved_problems
        )

        if rank == 0 and len(world_solved_problems) == world_num_problems:
            break
        elif rank != 0 and len(local_remaining_problems) == 0:
            break

        sync_toggle = False
        for problem in local_remaining_problems:
            search_result = np.zeros(8, dtype=np.int64)
            start_time = time.time()
            (solution_length, num_expanded, num_generated, traj,) = agent.search(
                problem,
                model,
                current_budget,
                update_levin_costs,
                train=False,
            )
            end_time = time.time()
            if is_bidirectional:
                problem.domain.reset()

            problem_id = problem[0]
            start_time = int(start_time * 1000)
            end_time = int(end_time * 1000)

            search_result[0] = problem_id
            search_result[1] = solution_length
            search_result[2] = current_budget
            search_result[3] = num_expanded
            search_result[4] = num_generated
            search_result[5] = start_time
            search_result[6] = end_time
            search_result[7] = end_time - start_time

            if traj:
                assert problem_id not in local_solved_problems
                local_solved_problems.add(problem_id)

            results_queue.put(search_result)
            if rank == 0:
                if sync_toggle:
                    num_expanded = try_sync_results()
                    if num_generated > 0:
                        sync_toggle = False
                        total_num_expanded += num_expanded
                else:
                    sync_toggle = True

        if rank == 0:
            total_num_expanded += try_sync_results()
        # epoch end
        current_budget *= 2

    # test end
    print(f"Rank {rank} finished")
    if rank == 0:
        world_results_df.to_csv(f"{writer.log_dir}/results.csv")
