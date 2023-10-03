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
    batch_size = args.world_size

    num_valid_problems = len(valid_loader.all_ids)
    max_valid_expanded = num_valid_problems * expansion_budget
    best_valid_solved = -1
    best_valid_total_expanded = max_valid_expanded + 1

    stage_flosses = []
    stage_faccs = []
    stage_blosses = []
    stage_baccs = []
    # for running avg of stage solve rate
    stage_avg = 0
    stage_problems_sovled = 0

    for epoch in range(1, args.epochs + 1):
        flosses = []
        faccs = []
        blosses = []
        baccs = []
        batches = []

        # for epoch search results
        ids = []
        times = []
        fexps = []
        bexps = []
        flens = []
        blens = []
        fpnlls = []
        bpnlls = []
        fnlls = []
        bnlls = []
        stages = []
        # num_new_problems_solved_this_epoch = 0
        epoch_solved = 0
        epoch_problems_seen = 0
        epoch_expanded = 0

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

        old_stage = train_loader.stage
        for problem, stage in train_loader:
            if stage != old_stage:
                old_stage = stage
                stage_flosses = []
                stage_faccs = []
                stage_blosses = []
                stage_baccs = []
                # for running avg of stage solve rate
                stage_avg = 0
                stage_problems_sovled = 0

            model.eval()
            to.set_grad_enabled(False)

            f_trajs = []
            b_trajs = []

            num_problems_solved_this_batch = 0

            start_time = timer()
            (
                n_forw_expanded,
                n_backw_expanded,
                _,
                _,
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
            local_search_results[5] = solution_length

            if traj:
                f_traj, b_traj = traj
                f_trajs.append(f_traj)
                local_search_results[6] = f_traj.partial_g_cost
                local_search_results[8] = -1 * f_traj.partial_log_prob
                local_search_results[10] = -1 * f_traj.log_prob
                if b_traj:
                    b_trajs.append(b_traj)
                    local_search_results[7] = b_traj.partial_g_cost
                    local_search_results[9] = -1 * b_traj.partial_log_prob
                    local_search_results[11] = -1 * b_traj.log_prob

            dist.all_gather(world_search_results, local_search_results)
            world_batch_results_t = to.cat(world_search_results, dim=0)

            batch_results_arr = world_batch_results_t.numpy()

            world_batch_ids = np.array(
                [train_loader.all_ids[i] for i in batch_results_arr[:, 0].astype(int)],
                dtype=np.unicode_,
            )

            batch_print_df = pd.DataFrame(
                batch_results_arr, columns=search_result_header[1:]
            )
            batch_print_df["Id"] = world_batch_ids
            batch_print_df = batch_print_df.set_index("Id")
            for col in int_columns:
                batch_print_df[col] = batch_print_df[col].astype(int)
            batch_print_df = batch_print_df.sort_values("Exp")

            batch_solved_ids = batch_print_df[batch_print_df["Len"] > 0]["Len"]
            # for problem_id in batch_solved_ids:
            #     if problem_id not in solved_problems:
            #         num_new_problems_solved_this_epoch += 1
            #         solved_problems.add(problem_id)

            num_problems_solved_this_batch = len(batch_solved_ids)
            epoch_solved += num_problems_solved_this_batch
            epoch_problems_seen += batch_size

            batch_expansions = batch_print_df["Exp"].sum()
            epoch_expanded += batch_expansions

            stage_problems_sovled += num_problems_solved_this_batch

            batch_avg = num_problems_solved_this_batch / batch_size
            batches_seen += 1
            stage_avg += (batch_avg - stage_avg) / (batches_seen)

            if rank == 0:
                print(f"\n\nBatch {batches_seen}")

            if rank == 0:
                print(
                    tabulate(
                        batch_print_df,
                        headers="keys",
                        tablefmt="psql",
                        # floatfmt=".2f"
                        # intfmt="",
                    )
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    batch_expansions_ratio = batch_expansions / (
                        len(batch_print_df) * expansion_budget
                    )
                    fb_exp_ratio = (
                        batch_print_df["FExp"] / batch_print_df["BExp"]
                    ).mean()
                    fb_g_ratio = (batch_print_df["Fg"] / batch_print_df["Bg"]).mean()

                print(f"{'Solved':23s}: {num_problems_solved_this_batch}/{batch_size}")
                print(f"{'Total expansion ratio':23s}: {batch_expansions_ratio:.3f}")
                print(f"{'F/B expansion ratio':23s}: {fb_exp_ratio:.3f}")
                print(f"{'F/B g-cost ratio':23s}: {fb_g_ratio:.3f}\n")

                # if batches_seen % param_log_interval == 0:
                #     log_params(writer, model, batches_seen)

            for i in range(batch_size):
                ids.append(batch_results_arr[i, 0])
                times.append(batch_results_arr[i, 1])
                fexps.append(batch_results_arr[i, 3])
                bexps.append(batch_results_arr[i, 4])
                flens.append(batch_results_arr[i, 6])
                blens.append(batch_results_arr[i, 7])
                fpnlls.append(batch_results_arr[i, 8])
                bpnlls.append(batch_results_arr[i, 9])
                fnlls.append(batch_results_arr[i, 10])
                bnlls.append(batch_results_arr[i, 11])
                stages.append(stage)

            # perform grad steps
            dist.all_reduce(local_opt_results, op=dist.ReduceOp.SUM)
            num_procs_found_solution = int(local_opt_results[2].item())

            if num_procs_found_solution > 0:
                to.set_grad_enabled(True)
                model.train()
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
                            ) = loss_fn(b_trajs, model)

                            local_opt_results[3] = b_avg_action_nll
                            local_opt_results[4] = b_acc

                            loss = f_loss + b_loss
                        else:
                            loss = f_loss

                        loss.backward()
                    else:
                        local_opt_results[:] = 0

                    if num_procs_found_solution > 0:
                        sync_grads(model, num_procs_found_solution)

                    optimizer.step()

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
                                    f"{f_loss:5.3f}  {f_acc:5.3f}    {b_loss:5.3f}  {b_acc:5.3f}"
                                )
                            else:
                                print(f"{f_loss:5.3f}  {f_acc:5.3f}")

                        if grad_step == grad_steps:
                            flosses.append(f_loss)
                            faccs.append(f_acc)
                            if bidirectional:
                                blosses.append(b_loss)
                                baccs.append(b_acc)
                            batches.append(batches_seen)
            # BATCH END

            if rank == 0:
                epoch_solved_ratio = epoch_solved / epoch_problems_seen
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
                    epoch_f_loss = flosses.mean(where=(flosses >= 0))
                    epoch_f_acc = faccs.mean(where=(faccs >= 0))
                    if bidirectional:
                        epoch_b_loss = blosses.mean(where=(blosses >= 0))
                        epoch_b_acc = baccs.mean(where=(baccs >= 0))
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
                    f"{'Solved':20s}: {epoch_solved}/{world_num_problems} {epoch_solved_ratio}\n"
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
