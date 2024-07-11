import sys
from timeit import default_timer as timer

import numpy as np
from tabulate import tabulate
import torch as to
import torch.distributed as dist
import torch.multiprocessing as mp

from search.agent import Agent
from search.loaders import ArrayLoader
from search.utils import Result, ResultsLog


def test(
    args,
    rank: int,
    agent: Agent,
    loader: ArrayLoader,
    results_queue: mp.Queue,
    print_results: bool = True,
):
    current_exp_budget: int = args.test_expansion_budget

    to.set_grad_enabled(False)
    agent.model.eval()

    solved_set = set()

    results = ResultsLog(agent)
    epoch_buffer: list[Result] = []
    epoch = 0
    done_epoch = True
    assert results_queue.empty()
    while True:
        if done_epoch:
            epoch += 1
            done_epoch = False
            if rank == 0:
                loader.reset_indices(shuffle=False)
                problem = loader.next_batch()
            dist.barrier()

        problem = loader.get()
        if problem is not None:
            if problem.id in solved_set:
                continue
            agent.model.eval()
            to.set_grad_enabled(False)

            start_time = timer()
            (
                n_forw_expanded,
                n_backw_expanded,
                (f_traj, b_traj),
            ) = agent.search(
                problem,
                current_exp_budget,
                time_budget=args.time_budget,
            )
            end_time = timer()

            if f_traj is not None:
                sol_len = len(f_traj)
            else:
                sol_len = np.nan

            res = Result(
                id=problem.id,
                time=end_time - start_time,
                fexp=n_forw_expanded,
                bexp=n_backw_expanded,
                len=sol_len,
                f_traj=f_traj,
                b_traj=b_traj,
            )
            results_queue.put(res)
            del res

        else:  # end epoch
            done_epoch = True
            dist.barrier()
            if rank == 0:
                for _ in range(len(loader)):
                    res = results_queue.get()
                    epoch_buffer.append(res)
                    del res

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
                    sys.stdout.flush()

            if args.increase_budget:
                if rank == 0:
                    for id in (p.id for p in epoch_buffer if p.len > 0):
                        solved_set.add(id)
                    tmp = [solved_set]
                else:
                    tmp = [None]
                dist.broadcast_object_list(tmp, 0)
                current_exp_budget *= 2
                epoch_buffer.clear()
                del tmp
            else:
                epoch_buffer.clear()
                break

    results_df = results.get_df()
    results.clear()
    del results, epoch_buffer
    dist.barrier()
    return results_df
