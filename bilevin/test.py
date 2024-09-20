import sys
from timeit import default_timer as timer

import numpy as np
import pickle as pkl
from pathlib import Path
from tabulate import tabulate
import torch as to
import torch.distributed as dist
import torch.multiprocessing as mp

from search.agent import Agent
from search.loaders import ArrayLoader
from search.utils import Result, ResultsLog


def test(
    rank: int,
    agent: Agent,
    loader: ArrayLoader,
    results_queue: mp.Queue,
    exp_budget: int,
    time_budget: int,
    print_results: bool = True,
    solved_results_path: Path | None = None,
    results_df_path: Path | None = None,
):
    assert results_queue.empty()

    to.set_grad_enabled(False)
    agent.model.eval()

    results_df = None

    if rank == 0:
        loader.reset_indices(shuffle=False)
        problem = loader.next_batch()
    dist.barrier()

    while True:
        problem = loader.get()
        if problem is not None:
            agent.model.eval()
            to.set_grad_enabled(False)

            start_time = timer()
            (
                n_forw_expanded,
                n_backw_expanded,
                (f_traj, b_traj),
            ) = agent.search(
                problem,
                exp_budget,
                time_budget=time_budget,
            )
            end_time = timer()

            if f_traj is not None:
                sol_len = len(f_traj)
            else:
                sol_len = np.nan

            exp = n_forw_expanded + n_backw_expanded
            res = Result(
                id=problem.id,
                time=end_time - start_time,
                exp=exp,
                len=sol_len,
                f_traj=f_traj,
                b_traj=b_traj,
            )
            results_queue.put(res)
            del res

        else:  # done test
            dist.barrier()
            break

    if rank == 0:
        all_results: list[Result] = []
        for _ in range(len(loader)):
            res = results_queue.get()
            all_results.append(res)
            del res

        results_log = ResultsLog(all_results)
        results_df = results_log.get_df()
        if print_results:
            print(f"\nBudget {exp_budget}")
            print(
                tabulate(
                    results_df,
                    headers="keys",
                    tablefmt="psql",
                    showindex=False,
                    floatfmt=".2f",
                )
            )
            sys.stdout.flush()
        results_log.clear()

        solved_results = []
        for res in all_results:
            if res.len > 0:
                solved_results.append(res)

        if solved_results_path is not None:
            with open(solved_results_path, "wb") as f:
                pkl.dump(solved_results, f)

        if results_df_path is not None:
            with open(results_df_path, "wb") as f:
                pkl.dump(results_df, f)

        del solved_results, results_log, all_results

    dist.barrier()
    return results_df
