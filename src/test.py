import csv
from pathlib import Path
from timeit import default_timer as timer
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from tabulate import tabulate
import torch as to
import torch.distributed as dist

from loaders import ProblemLoader
from search.agent import Agent
from search.utils import int_columns, search_result_header, test_csvfields


def test(
    args,
    rank: int,
    agent: Agent,
    problems_loader: ProblemLoader,
    print_results: bool = True,
    epoch: Optional[int] = None,
):
    current_exp_budget = args.expansion_budget
    current_time_budget = args.time_budget

    world_num_problems = len(problems_loader.all_ids)

    bidirectional = agent.bidirectional
    model = agent.model

    to.set_grad_enabled(False)
    if agent.trainable:
        model.eval()

    total_num_expanded = 0

    local_problems = set(p[0] for p in problems_loader if p)
    local_search_results = np.zeros(
        (len(local_problems), len(search_result_header)), dtype=np.float64
    )
    local_solved_problems = [False] * len(local_problems)

    if rank == 0:
        world_search_results = [None] * world_size
        test_csv = logdir / "test.csv"
        if test_csv.exists():
            test_csv = test_csv.open("a", newline="")
            test_writer = csv.DictWriter(test_csv, test_csvfields)
        else:
            test_csv = test_csv.open("w", newline="")
            test_writer = csv.DictWriter(test_csv, test_csvfields)
            test_writer.writeheader()
    else:
        world_search_results = None

    fb_exp_ratio = float("nan")
    fb_g_ratio = float("nan")
    avg_sol_len = float("nan")

    if rank == 0:
        print("Testing...")
    test_start_time = timer()
    while True:
        for i, problem in enumerate(tuple(local_problems)):
            if local_solved_problems[i]:
                continue

            start_time = timer()
            (
                n_forw_expanded,
                n_backw_expanded,
                n_forw_generated,
                n_backw_generated,
                traj,
            ) = agent.search(
                problem,
                current_exp_budget,
                time_budget=current_time_budget,
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
                local_solved_problems[i] = True

                local_search_results[i, 9] = traj[0].partial_g_cost
                local_search_results[i, 11] = -1 * traj[0].partial_log_prob
                local_search_results[i, 13] = -1 * traj[0].log_prob
                if bidirectional:
                    local_search_results[i, 10] = traj[1].partial_g_cost
                    local_search_results[i, 12] = -1 * traj[1].partial_log_prob
                    local_search_results[i, 14] = -1 * traj[1].log_prob
            local_search_results[i, 15] = end_time - test_start_time

        dist.barrier()

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

    dist.gather_object(local_search_results, world_search_results)

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
        world_results_df = world_results_df.sort_values("Exp")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fb_exp_ratio = (world_results_df["FExp"] / world_results_df["BExp"]).mean()
            fb_g_ratio = (world_results_df["Fg"] / world_results_df["Bg"]).mean()
            fb_pnll_ratio = (
                world_results_df["FPnll"] / world_results_df["BPnll"]
            ).mean()
            fb_nll_ratio = (
                world_results_df["FPnll"] / world_results_df["BPnll"]
            ).mean()
            avg_sol_len = world_results_df[world_results_df["Len"] > 0]["Len"].mean()

        if print_results:
            print(
                tabulate(
                    world_results_df,
                    headers="keys",
                    tablefmt="psql",
                )
            )
            print(f"{'Solved':23s}: {current_num_solved}/{world_num_problems}\n")
            print(f"{'F/B expansion ratio':23s}: {fb_exp_ratio:5.3f}")
            print(f"{'F/B g-cost ratio':23s}: {fb_g_ratio:5.3f}")
            print(f"{'Avg solution length':20s}: {avg_sol_len:5.3f}\n")

        total_num_expanded += world_results_df["Exp"].sum()

        print(
            f"Solved: {current_num_solved}/{world_num_problems} in {timer() - test_start_time:.2f}s"
        )
        if not epoch:
            this_epoch = 1
        else:
            this_epoch = epoch
        test_writer.writerow(
            {
                "epoch": this_epoch,
                "solved": current_num_solved,
                "sol_len": avg_sol_len,
                "exp_ratio": fb_exp_ratio,
                "fb_exp_ratio": fb_exp_ratio,
                "fb_g_ratio": fb_g_ratio,
                "fb_pnll_ratio": fb_pnll_ratio,
                "fb_nll_ratio": fb_nll_ratio,
            }
        )
        test_csv.flush()

    if rank == 0:
        return (
            current_num_solved,
            total_num_expanded,
            fb_exp_ratio,
            fb_g_ratio,
            avg_sol_len,
        )

    return None, None, None, None, None
