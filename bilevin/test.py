from pathlib import Path
import pickle
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from tabulate import tabulate
import torch as to
import torch.distributed as dist
import torch.multiprocessing as mp

from search.agent import Agent
from search.loaders import AsyncProblemLoader
from search.utils import Result, ResultsLog
from search.utils import int_columns, print_search_summary, search_result_header


def test(
    args,
    rank: int,
    agent: Agent,
    loader: AsyncProblemLoader,
    results_queue: mp.Queue,
    print_results: bool = True,
    increase_budget: bool = False,
    train_epoch: int = 0,
):
    print(f"rank {rank} batch size {loader.batch_size} problems {len(loader.problems)}")
    assert loader.batch_size == len(loader.problems)
    current_exp_budget: int = args.test_expansion_budget
    current_time_budget: float = args.time_budget

    to.set_grad_enabled(False)
    agent.model.eval()

    solved_set = set()

    results = ResultsLog(
        None, agent.has_policy, agent.has_heuristic, agent.is_bidirectional
    )

    epoch = 0
    done_epoch = True
    while True:
        if done_epoch:
            epoch += 1
            done_epoch = False
            if rank == 0:
                loader.init_indexer(shuffle=False)
            if rank == 0:
                problem = loader.advance_batch()
            dist.monitored_barrier()
            if rank != 0:
                problem = loader.get_problem()
        else:
            problem = loader.get_problem()

        if problem is not None and problem.id not in solved_set:
            agent.model.eval()
            to.set_grad_enabled(False)

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

            if traj is not None:
                sol_len = len(traj[0])
            else:
                sol_len = np.nan

            res = Result(
                id=problem.id,
                time=end_time - start_time,
                fexp=n_forw_expanded,
                bexp=n_backw_expanded,
                len=sol_len,
            )
            results_queue.put(res)

        else:  # end epoch
            done_epoch = True
            dist.monitored_barrier()
            if rank == 0:
                epoch_buffer: list[Result] = []
                while not results_queue.empty():
                    epoch_buffer.append(results_queue.get())

                start_idx = len(results)
                end_idx = len(epoch_buffer)
                results.append(epoch_buffer)
                epoch_df = results[start_idx:end_idx].get_df()
                if print_results:
                    print(f"\nEpoch {epoch}")
                    print(
                        tabulate(
                            epoch_df,
                            headers="keys",
                            tablefmt="psql",
                            showindex=False,
                            floatfmt=".2f",
                        )
                    )

            if increase_budget:
                if rank == 0:
                    for id in (p.id for p in epoch_buffer if p.len > 0):
                        solved_set.add(id)
                    tmp = [solved_set]
                else:
                    tmp = [None]
                dist.broadcast_object_list(tmp, 0)
                current_exp_budget *= 2
            else:
                break

    if rank == 0:
        print_search_summary(
            epoch_df,
            agent.is_bidirectional,
        )
        if train_epoch:
            pth = args.logdir / f"search_valid_e{train_epoch}.pkl"
        else:
            pth = args.logdir / f"test.pkl"

        with pth.open("wb") as f:
            pickle.dump(epoch_df, f)

    dist.monitored_barrier()
    return results
