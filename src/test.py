import math
import time
from typing import Callable, Type, Union

import numpy as np
import pandas as pd
from tabulate import tabulate
import torch as to
import torch.distributed as dist
from torch.utils.tensorboard.writer import SummaryWriter
import tqdm

from domains.domain import Problem
from train import ProblemsBatchLoader


def test(
    agent,
    model: Union[to.nn.Module, tuple[to.nn.Module, to.nn.Module]],
    problems: list[Problem],
    local_batch_size: int,
    writer: SummaryWriter,
    world_size: int,
    update_levin_costs: bool,
    initial_budget: int,
):
    current_budget = initial_budget
    dummy_last = False

    if world_size > 1:
        rank = dist.get_rank()

        sh_t = to.zeros(1, dtype=to.int32) + len(problems)
        dist.all_reduce(sh_t, dist.ReduceOp.SUM)
        world_num_problems = int(sh_t[0].item())

        n_batches = math.ceil(len(problems) / local_batch_size)
        sh_t[0] = n_batches
        dist.all_reduce(sh_t, dist.ReduceOp.MAX)
        if n_batches < sh_t[0]:
            dummy_last = True
    else:
        world_num_problems = len(problems)
        rank = 0

    world_batches_per_epoch = math.ceil(
        world_num_problems / (local_batch_size * world_size)
    )

    search_result_header = [
        "ProblemId",
        "SolutionLength",
        "NumExpanded",
        "NumGenerated",
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
    world_results_df.set_index("ProblemId", inplace=True)

    is_bidirectional = agent.bidirectional
    if is_bidirectional:
        assert isinstance(model, tuple)
        f_model, b_model = model

        for param in b_model.parameters():
            if not param.grad:
                param.grad = to.zeros_like(param)
    else:
        assert isinstance(model, to.nn.Module)
        f_model = model

    for param in f_model.parameters():
        if not param.grad:
            param.grad = to.zeros_like(param)

    problems_loader = ProblemsBatchLoader(
        problems, batch_size=local_batch_size, shuffle=False, dummy_last=dummy_last
    )
    world_batch_results = [
        to.zeros((local_batch_size, 5), dtype=to.int32) for _ in range(world_size)
    ]

    batches_seen = 0
    epoch = 0

    solved_problems = set()

    while len(solved_problems) < world_num_problems:
        epoch += 1

        num_new_problems_solved_this_epoch = 0
        num_problems_solved_this_epoch = 0

        if rank == 0:
            print(
                "============================================================================"
            )
            print(f"\nBeginning epoch {epoch} with budget {current_budget}")
            print(
                "============================================================================\n"
            )

        if rank == 0:
            problems_loader = tqdm.tqdm(problems_loader, total=world_batches_per_epoch)

        for local_batch_problems in problems_loader:
            batches_seen += 1

            to.set_grad_enabled(False)
            f_model.eval()
            if is_bidirectional:
                b_model.eval()  # type:ignore

            num_problems_solved_this_batch = 0
            local_batch_search_results = to.zeros(local_batch_size, 5, dtype=to.int32)
            for i, problem in enumerate(local_batch_problems):
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

                # todo handle single state trajectories, they break batchnorm layers
                local_batch_search_results[i, 0] = problem[0]
                local_batch_search_results[i, 1] = solution_length
                local_batch_search_results[i, 2] = num_expanded
                local_batch_search_results[i, 3] = num_generated
                local_batch_search_results[i, 4] = int((end_time - start_time) * 1000)

            if world_size > 1:
                dist.all_gather(world_batch_results, local_batch_search_results)
                world_batch_results_t = to.cat(world_batch_results, dim=0)
            else:
                world_batch_results_t = local_batch_search_results

            # hacky way to filter out results from partial batches
            world_batch_results_arr = world_batch_results_t.numpy()
            world_batch_results_arr = world_batch_results_arr[
                world_batch_results_arr[:, 2] > 0
            ]

            world_batch_ids = world_batch_results_arr[:, 0]
            world_results_df.loc[
                world_batch_ids, search_result_header[1:-1]
            ] = world_batch_results_arr[
                :, 1:-1
            ]  # ProblemId is already index, Time is set in following lines
            world_results_df.loc[world_batch_ids, "Time"] = (
                world_batch_results_arr[:, -1].astype(float) / 1000
            )

            world_batch_results_df = world_results_df.loc[world_batch_ids]
            world_batch_results_df.sort_values("NumExpanded", inplace=True)

            batch_solved_ids = world_batch_ids[world_batch_results_arr[:, 1] > 0]
            for problem_id in batch_solved_ids:
                if problem_id not in solved_problems:
                    num_new_problems_solved_this_epoch += 1
                    solved_problems.add(problem_id)

            num_problems_solved_this_batch = len(batch_solved_ids)
            num_problems_solved_this_epoch += num_problems_solved_this_batch
            num_problems_this_batch = len(world_batch_results_arr)

            if rank == 0:
                print(
                    tabulate(
                        world_batch_results_df,
                        headers="keys",
                        tablefmt="psql",
                    )
                )
                print(
                    f"Solved {num_problems_solved_this_batch}/{num_problems_this_batch}\n"
                )
                writer.add_scalar(
                    f"cum_unique_solved_vs_batch", len(solved_problems), batches_seen
                )

        if rank == 0:
            epoch_solved = num_problems_solved_this_epoch / world_num_problems
            print(
                "============================================================================"
            )
            print(
                f"Completed epoch {epoch}, solved {num_problems_solved_this_epoch}/{world_num_problems} problems with budget {current_budget}\n"
                f"Solved {num_new_problems_solved_this_epoch} new problems, {world_num_problems - len(solved_problems)} remaining\n"
            )
            print(
                "============================================================================\n"
            )

            # fmt: off
            writer.add_scalar("budget_vs_epoch", current_budget, epoch)
            writer.add_scalar(f"budget_{current_budget}/solved_vs_epoch", epoch_solved, epoch)
            writer.add_scalar("cum_unique_solved_vs_epoch", len(solved_problems), epoch)

            world_results_df.to_csv(f"{writer.log_dir}/epoch_{epoch}.csv")
            # fmt: on

        current_budget *= 2
