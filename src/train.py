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

import csv
from math import ceil
from pathlib import Path
import sys
from timeit import default_timer as timer
import warnings

import numpy as np
import pandas as pd
from tabulate import tabulate
from test import test
import torch as to
import torch.distributed as dist
from tqdm import tqdm

from loaders import ProblemLoader
from search.agent import Agent
from search.utils import Problem
from search.utils import int_columns, search_result_header, train_csvfields


def train(
    args,
    rank: int,
    agent: Agent,
    train_problems: list[list[Problem]],
    train_all_ids: list[str],
    valid_problems: list[list[Problem]],
    valid_all_ids: list[str],
    seed: int,
):
    if rank == 0:
        train_csv = (args.logdir / "train.csv").open("w", newline="")
        train_writer = csv.DictWriter(train_csv, train_csvfields)
        train_writer.writeheader()

        best_csv = (args.logdir / "best_models.csv").open("w", newline="")
        best_writer = csv.DictWriter(best_csv, ["epoch", "solve_rate", "exp_ratio"])
        best_writer.writeheader()

    opt_result_header = (
        f"           Forward        Backward\nOptStep   Loss    Acc    Loss    Acc"
    )

    bidirectional = agent.bidirectional
    model = agent.model
    optimizer = agent.optimizer
    loss_fn = agent.loss_fn

    expansion_budget = args.expansion_budget
    grad_steps = args.grad_steps

    for param in model.parameters():
        if not param.grad:
            param.grad = to.zeros_like(param)

    train_loader = ProblemLoader(train_problems, train_all_ids, seed=seed)
    valid_loader = ProblemLoader(valid_problems, valid_all_ids, seed=seed)

    local_opt_results = to.zeros(5, dtype=to.float64)

    local_search_results = to.zeros(len(search_result_header), dtype=to.float64)
    world_search_results = [
        to.zeros(len(search_result_header), dtype=to.float64)
        for _ in range(args.world_size)
    ]

    batches_seen = 0
    solved_problems = set()
    total_num_expanded = 0
    opt_step = 1
    opt_passes = 1

    num_valid_problems = len(valid_loader.all_ids)
    max_valid_expanded = num_valid_problems * expansion_budget
    best_valid_solved = -1
    best_valid_total_expanded = max_valid_expanded + 1

    for epoch in range(1, args.epochs + 1):
        world_epoch_f_loss = []
        world_epoch_f_acc = []
        world_epoch_b_loss = []
        world_epoch_b_acc = []
        # for running avg of stage solve rate
        stage_avg = 0
        stage_problems_sovled = 0

        if rank == 0:
            print(
                "============================================================================"
            )
            print(f"START | STAGE {train_loader.stage} EPOCH {epoch}")

        if epoch == args.epoch_reduce_lr:
            new_lr = optimizer.param_groups[0]["lr"] * 0.1
            if rank == 0:
                print(
                    f"-> Learning rate reduced from {optimizer.param_groups[0]['lr']} to {new_lr}"
                )

            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr

        if epoch == args.epoch_reduce_grad_steps:
            old_gs = grad_steps
            grad_steps = grad_steps // 2
            if rank == 0:
                print(f"-> Grad steps reduced from {old_gs} to {grad_steps}")
        if rank == 0:
            print(
                "============================================================================\n"
            )

        epoch_start_time = timer()

        for problem in train_loader:
            num_new_problems_solved_this_epoch = 0
            num_problems_solved_this_epoch = 0

            batches_seen += 1

            if rank == 0:
                print(f"\n\nBatch {batches_seen}")

            model.eval()
            to.set_grad_enabled(False)

            f_trajs = []
            b_trajs = []

            num_problems_solved_this_batch = 0

            start_time = timer()
            (
                n_forw_expanded,
                n_backw_expanded,
                n_forw_generated,
                n_backw_generated,
                traj,
            ) = agent.search(
                problem,
                expansion_budget,
                time_budget=time_budget,
            )
            end_time = timer()
            solution_length = 0 if not traj else traj[0].cost

            if bidirectional:
                problem.domain.reset()

            local_search_results[0] = problem.id_idx
            local_search_results[1] = end_time - start_time
            local_search_results[2] = n_forw_expanded + n_backw_expanded
            local_search_results[3] = n_forw_expanded
            local_search_results[4] = n_backw_expanded
            local_search_results[5] = n_forw_generated + n_backw_generated
            local_search_results[6] = n_forw_generated
            local_search_results[7] = n_backw_generated
            local_search_results[8] = solution_length

            if traj:
                f_traj, b_traj = traj
                f_trajs.append(f_traj)
                local_search_results[9] = f_traj.partial_g_cost
                local_search_results[11] = -1 * f_traj.partial_log_prob
                local_search_results[13] = -1 * f_traj.log_prob
                if b_traj:
                    b_trajs.append(b_traj)
                    local_search_results[10] = b_traj.partial_g_cost
                    local_search_results[12] = -1 * b_traj.partial_log_prob
                    local_search_results[14] = -1 * b_traj.log_prob

            local_search_results[15] = end_time - epoch_start_time

            dist.all_gather(world_search_results, local_search_results)
            world_batch_results_t = to.cat(world_search_results, dim=0)

            world_batch_results_arr = world_batch_results_t.numpy()
            # results with no expanded nodes are not valid (from a partial batch)
            world_batch_results_arr = world_batch_results_arr[
                world_batch_results_arr[:, 2] > 0
            ]

            world_batch_ids = np.array(
                [
                    train_loader.all_ids[i]
                    for i in world_batch_results_arr[:, 0].astype(int)
                ],
                dtype=np.unicode_,
            )

            world_batch_print_df = world_batch_results
            for col in int_columns:
                world_batch_print_df[col] = world_batch_print_df[col].astype(int)
            world_batch_print_df = world_batch_print_df.sort_values("Exp")

            batch_solved_ids = world_batch_ids[world_batch_results_arr[:, 8] > 0]
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
                        world_batch_print_df,
                        headers="keys",
                        tablefmt="psql",
                        # floatfmt=".2f"
                        # intfmt="",
                    )
                )

                batch_avg = num_problems_solved_this_batch / num_problems_this_batch

                batch_expansions = world_batch_print_df["Exp"].sum()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    batch_expansions_ratio = batch_expansions / (
                        len(world_batch_print_df) * expansion_budget
                    )
                    fb_exp_ratio = (
                        world_batch_print_df["FExp"] / world_batch_print_df["BExp"]
                    ).mean()
                    fb_g_ratio = (
                        world_batch_print_df["Fg"] / world_batch_print_df["Bg"]
                    ).mean()

                print(
                    f"{'Solved':23s}: {num_problems_solved_this_batch}/{num_problems_this_batch}"
                )
                print(f"{'Total expansion ratio':23s}: {batch_expansions_ratio:.3f}")
                print(f"{'F/B expansion ratio':23s}: {fb_exp_ratio:.3f}")
                print(f"{'F/B g-cost ratio':23s}: {fb_g_ratio:.3f}\n")

                total_num_expanded += world_batch_print_df["Exp"].sum()
                # if batches_seen % param_log_interval == 0:
                #     log_params(writer, model, batches_seen)

            # perform grad steps
            to.set_grad_enabled(True)
            model.train()

            num_procs_found_solution = 0
            f_loss = -1
            f_acc = -1
            b_loss = -1
            b_acc = -1

            for grad_step in range(1, grad_steps + 1):
                optimizer.zero_grad(set_to_none=False)
                if f_trajs:
                    (
                        f_loss,
                        f_avg_action_nll,
                        f_acc,
                    ) = loss_fn(f_trajs, model)

                    local_opt_results[0] = f_avg_action_nll
                    local_opt_results[1] = f_acc
                    local_opt_results[2] = 1

                    if bidirectional:
                        (
                            b_loss,
                            b_avg_action_nll,
                            b_acc,
                        ) = loss_fn(b_trajs, model, n_subgoals=n_subgoals)

                        local_opt_results[3] = b_avg_action_nll
                        local_opt_results[4] = b_acc

                        loss = f_loss + b_loss
                    else:
                        loss = f_loss

                    loss.backward()
                else:
                    local_opt_results[:] = 0

                dist.all_reduce(local_opt_results, op=dist.ReduceOp.SUM)
                num_procs_found_solution = int(local_opt_results[2].item())
                if num_procs_found_solution > 0:
                    sync_grads(model, num_procs_found_solution)

                optimizer.step()

                if num_procs_found_solution > 0:
                    if rank == 0:
                        if grad_step == 1 or grad_step == grad_steps:
                            if grad_step == 1:
                                print(opt_result_header)

                            f_loss = (
                                local_opt_results[0].item() / num_procs_found_solution
                            )
                            f_acc = (
                                local_opt_results[1].item() / num_procs_found_solution
                            )
                            b_loss = (
                                local_opt_results[3].item() / num_procs_found_solution
                            )
                            b_acc = (
                                local_opt_results[4].item() / num_procs_found_solution
                            )
                            if bidirectional:
                                print(
                                    f"{opt_step:7}  {f_loss:5.3f}  {f_acc:5.3f}    {b_loss:5.3f}  {b_acc:5.3f}"
                                )
                            else:
                                print(f"{opt_step:7}  {f_loss:5.3f}  {f_acc:5.3f}")

                    opt_step += 1
            if num_procs_found_solution > 0:
                opt_passes += 1

            world_epoch_f_loss[batch_idx] = f_loss
            world_epoch_f_acc[batch_idx] = f_acc
            if bidirectional:
                world_epoch_b_loss[batch_idx] = b_loss
                world_epoch_b_acc[batch_idx] = b_acc
            if rank == 0:
                print(
                    tqdm.format_meter(
                        n=batch_idx + 1,
                        total=world_batches_this_difficulty,
                        elapsed=timer() - epoch_start_time,
                    )
                )
                sys.stdout.flush()
            # BATCH END

            if rank == 0:
                epoch_expansions = world_results_df["Exp"].sum()
                epoch_solved_ratio = num_problems_solved_this_epoch / world_num_problems
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    epoch_expansions_ratio = epoch_expansions / max_epoch_expansions
                    epoch_avg_sol_len = world_results_df[world_results_df["Len"] > 0][
                        "Len"
                    ].mean()
                    epoch_fb_exp_ratio = (
                        world_results_df["FExp"] / world_results_df["BExp"]
                    ).mean()
                    epoch_fb_g_ratio = (
                        world_results_df["Fg"] / world_results_df["Bg"]
                    ).mean()
                    epoch_fb_pnll_ratio = (
                        world_results_df["FPnll"] / world_results_df["BPnll"]
                    ).mean()
                    epoch_fb_nll_ratio = (
                        world_results_df["FPnll"] / world_results_df["BPnll"]
                    ).mean()
                    epoch_f_loss = world_epoch_f_loss.mean(
                        where=(world_epoch_f_loss >= 0)
                    )
                    epoch_f_acc = world_epoch_f_acc.mean(where=(world_epoch_f_acc >= 0))
                    if bidirectional:
                        epoch_b_loss = world_epoch_b_loss.mean(
                            where=(world_epoch_b_loss >= 0)
                        )
                        epoch_b_acc = world_epoch_b_acc.mean(
                            where=(world_epoch_b_acc >= 0)
                        )
                print(
                    "============================================================================"
                )
                print(
                    f"END | STAGE {train_loader.stage} EPOCH {train_loader.stage_epoch} | TOTAL EPOCH {epoch}"
                )
                print(
                    "----------------------------------------------------------------------------"
                )
                print(
                    f"{'Solved':20s}: {num_problems_solved_this_epoch}/{world_num_problems} {epoch_solved_ratio}\n"
                    f"{'Expansions':20s}: {int(epoch_expansions)}/{max_epoch_expansions}  {epoch_expansions_ratio:5.3f}\n"
                    f"{'FB Exp Ratio':20s}: {epoch_fb_exp_ratio:5.3f}\n"
                    f"{'FB G-Cost Ratio':20s}: {epoch_fb_g_ratio:5.3f}\n"
                    f"{'Avg solution length':20s}: {epoch_avg_sol_len:5.3f}\n"
                )
                print(f"  Forward        Backward\nLoss    Acc    Loss    Acc")
                if bidirectional:
                    print(
                        f"{epoch_f_loss:5.3f}  {epoch_f_acc:5.3f}    {epoch_b_loss:5.3f}  {epoch_b_acc:5.3f}"
                    )
                else:
                    print(f"{epoch_f_loss:5.3f}  {epoch_f_acc:5.3f}")
                print(
                    "============================================================================"
                )
                train_writer.writerow(
                    {
                        "epoch": epoch,
                        "floss": epoch_f_loss,
                        "facc": epoch_f_acc,
                        "bloss": epoch_b_loss,
                        "bacc": epoch_b_acc,
                        "solved": epoch_solved_ratio,
                        "exp_ratio": epoch_expansions_ratio,
                        "fb_exp_ratio": epoch_fb_exp_ratio,
                        "fb_g_ratio": epoch_fb_g_ratio,
                        "sol_len": epoch_avg_sol_len,
                        "fb_pnll_ratio": epoch_fb_pnll_ratio,
                        "fb_nll_ratio": epoch_fb_nll_ratio,
                    }
                )
                train_csv.flush()
                # fmt: on

            if epoch >= epoch_begin_validate and epoch % validate_every == 0:
                if rank == 0:
                    print("VALIDATION")

                dist.barrier()

                valid_results = test(
                    rank,
                    args.logdir,
                    agent,
                    valid_loader,
                    args.world_size,
                    expansion_budget,
                    time_budget,
                    increase_budget=False,
                    print_results=False,
                    epoch=epoch,
                )

                if rank == 0:
                    (
                        valid_solved,
                        valid_total_expanded,
                        valid_fb_exp_ratio,
                        valid_fb_g_ratio,
                        valid_avg_sol_len,
                    ) = valid_results
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        valid_expansions_ratio = (
                            valid_total_expanded / max_valid_expanded
                        )
                    valid_solve_rate = valid_solved / num_valid_problems
                    print(
                        f"{'Solved':20s}:  {valid_solved}/{num_valid_problems} {valid_solve_rate:5.3f}"
                    )
                    print(
                        f"{'Expansions':20s}: {int(valid_total_expanded)}/{max_valid_expanded} {valid_expansions_ratio:5.3f}"
                    )
                    print(f"{'FB Exp Ratio':20s}: {valid_fb_exp_ratio:5.3f}")
                    print(f"{'FB G-Cost Ratio':20s}: {valid_fb_g_ratio:5.3f}")
                    # writer.add_scalar(f"budget_{budget}/valid_solve_rate", valid_solve_rate, epoch)
                    agent.save_model("latest", log=False)
                    if valid_total_expanded <= best_valid_total_expanded:
                        best_valid_total_expanded = valid_total_expanded
                        print("Saving best model by expansions")
                        best_writer.writerow(
                            {
                                "epoch": epoch,
                                "solve_rate": valid_solve_rate,
                                "exp_ratio": valid_expansions_ratio,
                            }
                        )
                        agent.save_model("best_expanded", log=False)

                    if valid_solved >= best_valid_solved:
                        best_valid_solved = valid_solved
                        print("Saving best model by solved")
                        best_writer.writerow(
                            {
                                "epoch": epoch,
                                "solve_rate": valid_solve_rate,
                                "exp_ratio": valid_expansions_ratio,
                            }
                        )
                        agent.save_model("best_solved", log=False)
                    best_csv.flush()
                    sys.stdout.flush()

                dist.barrier()

            epoch += 1
        # EPOCHS END
    if rank == 0:
        print("END TRAINING")


def sync_grads(model: to.nn.Module, n: int):
    all_grads_list = [param.grad.view(-1) for param in model.parameters()]
    all_grads = to.cat(all_grads_list)
    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
    all_grads.div_(n)
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.grad.data.copy_(
            all_grads[offset : offset + numel].view_as(param.grad.data)
        )
        offset += numel
