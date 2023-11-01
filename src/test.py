import pickle
from timeit import default_timer as timer
from typing import Optional

import numpy as np
import pandas as pd
from tabulate import tabulate
import torch as to
import torch.distributed as dist

from loaders import ProblemLoader
from search.agent import Agent
from search.utils import print_search_summary
from search.utils import int_columns, search_result_header


def test(
    args,
    rank: int,
    agent: Agent,
    problems_loader: ProblemLoader,
    print_results: bool = True,
    batch: Optional[int] = None,
):
    current_exp_budget = args.test_expansion_budget
    current_time_budget = args.time_budget
    world_size = args.world_size
    increase_budget = False
    logdir = args.logdir

    world_num_problems = len(problems_loader)

    bidirectional = agent.bidirectional
    model = agent.model

    to.set_grad_enabled(False)
    if agent.trainable:
        model.eval()

    total_num_expanded = 0

    local_problems = problems_loader.problems[0]  # test/valid problems have one stage
    local_search_results = np.zeros(
        (len(local_problems), len(search_result_header)), dtype=np.float64
    )
    local_search_results[:, :] = np.nan
    local_solved_problems = [False] * len(local_problems)

    world_search_results = [None] * world_size

    if rank == 0:
        print("Testing...")
    test_start_time = timer()
    while True:
        for i, problem in enumerate(local_problems):
            if local_solved_problems[i]:
                continue

            start_time = timer()
            (
                n_forw_expanded,
                n_backw_expanded,
                traj,
            ) = agent.search(
                problem,
                current_exp_budget,
                time_budget=current_time_budget,
            )
            end_time = timer()
            solution_length = 0 if not traj else len(traj[0])

            if bidirectional:
                problem.domain.reset()

            local_search_results[i, 0] = problem.id
            local_search_results[i, 1] = end_time - start_time
            local_search_results[i, 2] = n_forw_expanded + n_backw_expanded
            local_search_results[i, 3] = n_forw_expanded
            local_search_results[i, 4] = n_backw_expanded
            local_search_results[i, 5] = solution_length

            if traj:
                f_traj, b_traj = traj
                local_solved_problems[i] = True
                local_search_results[i, 6] = f_traj.partial_g_cost
                local_search_results[i, 8] = -1 * f_traj.partial_log_prob
                if b_traj:
                    local_search_results[i, 7] = b_traj.partial_g_cost
                    local_search_results[i, 9] = -1 * b_traj.partial_log_prob

        dist.monitored_barrier()

        num_solved_t = to.zeros(1, dtype=to.int64)
        num_solved_t[0] = sum(local_solved_problems)

        dist.all_reduce(num_solved_t, op=dist.ReduceOp.SUM)

        current_num_solved = num_solved_t.item()
        if current_num_solved == world_num_problems or not increase_budget:
            break

        if increase_budget:
            current_exp_budget *= 2
            current_time_budget *= 2
            if rank == 0:
                print(
                    f"Solved {current_num_solved}/{world_num_problems} problems\n"
                    f"Expandion budget increased from {int(current_exp_budget / 2)} to {int(current_exp_budget)}\n"
                    f"Time budget increased from {current_time_budget / 2} to {current_time_budget}\n"
                )

    dist.gather_object(
        local_search_results, world_search_results if rank == 0 else None
    )

    if rank == 0:
        world_search_results_arr = np.vstack(world_search_results)

        world_results_df = pd.DataFrame(
            world_search_results_arr, columns=search_result_header
        )
        for col in int_columns:
            world_results_df[col] = world_results_df[col].astype(int)

        world_results_df = world_results_df.sort_values("exp")

        if print_results:
            print(
                tabulate(
                    world_results_df,
                    headers="keys",
                    tablefmt="psql",
                    showindex=False,
                )
            )
        total_num_expanded += world_results_df["exp"].sum()

        stage_search_df = pd.DataFrame(
            {
                "id": world_results_df["id"].astype(pd.UInt32Dtype()),
                "time": world_results_df["time"].astype(pd.Float32Dtype()),
                "len": world_results_df["len"].astype(pd.UInt16Dtype()),
                "fexp": world_results_df["fexp"].astype(pd.UInt32Dtype()),
                "fg": world_results_df["fg"].astype(pd.UInt32Dtype()),
                "fpnll": world_results_df["fpnll"].astype(pd.Float32Dtype()),
            }
        )
        if bidirectional:
            stage_search_df["bexp"] = world_results_df["bexp"].astype(pd.UInt32Dtype())
            stage_search_df["bg"] = world_results_df["bg"].astype(pd.UInt16Dtype())
            stage_search_df["bpnll"] = world_results_df["bpnll"].astype(
                pd.Float32Dtype()
            )

        print_search_summary(stage_search_df, bidirectional)
        print(f"\nTime: {timer() - test_start_time:.2f}s")
        if not batch:
            pth = logdir / f"test.pkl"
        else:
            pth = logdir / f"search_valid_b{batch}.pkl"

        with pth.open("wb") as f:
            pickle.dump(stage_search_df, f)

    if rank == 0:
        return (
            current_num_solved,
            total_num_expanded,
        )

    return None, None
