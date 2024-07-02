from timeit import default_timer as timer

import numpy as np
from tabulate import tabulate
import torch as to
import torch.distributed as dist
import torch.multiprocessing as mp

from search.agent import Agent
from search.loaders import AsyncProblemLoader
from search.utils import Result, ResultsLog


def test(
    args,
    rank: int,
    agent: Agent,
    loader: AsyncProblemLoader,
    results_queue: mp.Queue,
    print_results: bool = True,
):
    assert loader.batch_size == len(loader.problems)
    current_exp_budget: int = args.test_expansion_budget
    current_time_budget: float = args.time_budget

    to.set_grad_enabled(False)
    agent.model.eval()

    solved_set = set()

    results = ResultsLog(None, agent)

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

                results.append(epoch_buffer)
                epoch_df = results[-len(epoch_buffer) :].get_df()
                if print_results:
                    print(f"\nBudget {current_exp_budget}")
                    print(
                        tabulate(
                            epoch_df,
                            headers="keys",
                            tablefmt="psql",
                            showindex=False,
                            floatfmt=".2f",
                        )
                    )

            if args.increase_budget:
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

    dist.monitored_barrier()
    return results.get_df()
