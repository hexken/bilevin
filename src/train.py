# offlini Copyright (C) 2021-2022, Ken Tjhia
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

from math import ceil
from pathlib import Path
from shutil import copyfile
import sys
from timeit import default_timer as timer
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from tabulate import tabulate
from test import test
import torch as to
import torch.distributed as dist
from torch.utils.tensorboard.writer import SummaryWriter
import tqdm

from loaders import CurriculumLoader, ProblemsBatchLoader
from search.agent import Agent
from search.utils import int_columns, search_result_header, get_subgoal_trajs


def train(
    rank: int,
    agent: Agent,
    train_loader: CurriculumLoader,
    writer: SummaryWriter,
    world_size: int,
    expansion_budget: int,
    time_budget: int,
    seed: int,
    grad_steps: int = 10,
    use_subgoal_trajs: bool = False,
    epoch_reduce_lr: int = 99999,
    epoch_reduce_grad_steps: int = 99999,
    epoch_begin_validate: int = 1,
    valid_loader: Optional[ProblemsBatchLoader] = None,
):
    is_distributed = world_size > 1

    if rank == 0:
        logdir = Path(writer.log_dir)

    opt_result_header = (
        f"           Forward        Backward\nOptStep   Loss    Acc    Loss    Acc"
    )

    bidirectional = agent.bidirectional
    model = agent.model
    optimizer = agent.optimizer
    loss_fn = agent.loss_fn

    for param in model.parameters():
        if not param.grad:
            param.grad = to.zeros_like(param)

    # if rank == 0:
    #     log_params(writer, model, 0)

    local_batch_opt_results = to.zeros(5, dtype=to.float64)

    local_batch_size = train_loader.local_batch_size
    local_batch_search_results = to.zeros(
        local_batch_size, len(search_result_header), dtype=to.float64
    )
    world_batch_search_results = [
        to.zeros((local_batch_size, len(search_result_header)), dtype=to.float64)
        for _ in range(world_size)
    ]

    batches_seen = 0
    solved_problems = set()
    total_num_expanded = 0
    opt_step = 1
    opt_passes = 1

    num_valid_problems = 0 if not valid_loader else len(valid_loader.all_ids)
    max_valid_expanded = num_valid_problems * expansion_budget
    best_valid_solved = -1
    best_valid_total_expanded = max_valid_expanded + 1

    epoch = 1
    train_start_time = timer()
    for batch_loader in train_loader:
        all_ids = batch_loader.all_ids
        world_num_problems = len(all_ids)
        if world_num_problems == 0:
            continue
        max_epoch_expansions = world_num_problems * expansion_budget

        world_batches_this_difficulty = ceil(
            world_num_problems / (local_batch_size * world_size)
        )

        dummy_data = np.column_stack(
            (
                np.zeros(
                    (world_num_problems, len(search_result_header) - 1),
                    dtype=np.int64,
                ),
            )
        )
        world_results_df = pd.DataFrame(dummy_data, columns=search_result_header[1:])
        del dummy_data
        world_results_df["ProblemId"] = all_ids
        world_results_df.set_index("ProblemId", inplace=True)
        world_results_df["Stage"] = train_loader.stage

        world_epoch_f_loss = np.zeros(world_batches_this_difficulty)
        world_epoch_f_acc = np.zeros(world_batches_this_difficulty)
        world_epoch_b_loss = np.zeros(world_batches_this_difficulty)
        world_epoch_b_acc = np.zeros(world_batches_this_difficulty)

        for stage_epoch in range(1, batch_loader.epochs + 1):
            num_new_problems_solved_this_epoch = 0
            num_problems_solved_this_epoch = 0
            # world_results_df["Epoch"] = epoch
            world_results_df["StageEpoch"] = stage_epoch
            world_results_df["Batch"] = 0

            if rank == 0:
                epoch_logdir = Path(writer.log_dir) / f"epoch-{epoch}"
                epoch_logdir.mkdir(parents=True, exist_ok=True)
                print(
                    "============================================================================"
                )
                print(
                    f"START | STAGE {train_loader.stage} EPOCH {stage_epoch} | TOTAL EPOCH {epoch}"
                )

            if epoch == epoch_reduce_lr:
                new_lr = optimizer.param_groups[0]["lr"] * 0.1
                if rank == 0:
                    print(
                        f"-> Learning rate reduced from {optimizer.param_groups[0]['lr']} to {new_lr}"
                    )

                for param_group in optimizer.param_groups:
                    param_group["lr"] = new_lr

            if epoch == epoch_reduce_grad_steps:
                old_gs = grad_steps
                grad_steps = grad_steps // 2
                if rank == 0:
                    print(f"-> Grad steps reduced from {old_gs} to {grad_steps}")
            if rank == 0:
                print(
                    "============================================================================\n"
                )
                batch_loader = tqdm.tqdm(
                    batch_loader, total=world_batches_this_difficulty
                )

            for batch_idx, local_batch_problems in enumerate(batch_loader):
                batches_seen += 1
                local_batch_search_results[
                    :
                ] = 0  # since final batch might contain <= local_batch_size problems

                if rank == 0:
                    print(f"\n\nBatch {batches_seen}")

                model.eval()
                to.set_grad_enabled(False)

                f_trajs = []
                b_trajs = []

                num_problems_solved_this_batch = 0
                for i, problem in enumerate(local_batch_problems):
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
                        train=True,
                    )
                    end_time = timer()
                    solution_length = 0 if not traj else traj[0].cost

                    if bidirectional:
                        problem.domain.reset()

                    local_batch_search_results[i, 0] = problem.id_idx
                    local_batch_search_results[i, 1] = end_time - start_time
                    local_batch_search_results[i, 2] = (
                        n_forw_expanded + n_backw_expanded
                    )
                    local_batch_search_results[i, 3] = n_forw_expanded
                    local_batch_search_results[i, 4] = n_backw_expanded
                    local_batch_search_results[i, 5] = (
                        n_forw_generated + n_backw_generated
                    )
                    local_batch_search_results[i, 6] = n_forw_generated
                    local_batch_search_results[i, 7] = n_backw_generated
                    local_batch_search_results[i, 8] = solution_length

                    if traj:
                        f_traj, b_traj = traj
                        f_trajs.append(f_traj)
                        local_batch_search_results[i, 9] = f_traj.partial_g_cost
                        local_batch_search_results[i, 11] = -1 * f_traj.partial_log_prob
                        local_batch_search_results[i, 13] = -1 * f_traj.log_prob
                        if bidirectional:
                            b_trajs.append(traj[1])
                            local_batch_search_results[i, 10] = b_traj.partial_g_cost
                            local_batch_search_results[i, 12] = (
                                -1 * b_traj.partial_log_prob
                            )
                            local_batch_search_results[i, 14] = -1 * b_traj.log_prob

                            if use_subgoal_trajs:
                                b_trajs.extend(get_subgoal_trajs(b_traj))

                    local_batch_search_results[i, 15] = end_time - train_start_time

                if is_distributed:
                    dist.all_gather(
                        world_batch_search_results, local_batch_search_results
                    )
                    world_batch_results_t = to.cat(world_batch_search_results, dim=0)
                else:
                    world_batch_results_t = local_batch_search_results

                world_batch_results_arr = world_batch_results_t.numpy()
                # results with no expanded nodes are not valid (from a partial batch)
                world_batch_results_arr = world_batch_results_arr[
                    world_batch_results_arr[:, 2] > 0
                ]

                world_batch_ids = np.array(
                    [all_ids[i] for i in world_batch_results_arr[:, 0].astype(int)],
                    dtype=np.unicode_,
                )
                world_results_df.loc[
                    world_batch_ids, search_result_header[1:]
                ] = world_batch_results_arr[
                    :, 1:
                ]  # ProblemId is already index
                world_results_df.loc[world_batch_ids, "Batch"] = batches_seen
                # done updating world_results_df

                world_batch_print_df = world_results_df.loc[
                    world_batch_ids, search_result_header[1:-1]
                ]
                for col in int_columns:
                    world_batch_print_df[col] = world_batch_print_df[col].astype(int)
                world_batch_print_df.sort_values("Exp")

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
                    writer.add_scalar(f"solved_vs_batch", batch_avg, batches_seen)

                    batch_expansions = world_batch_print_df["Exp"].sum()
                    batch_expansions_ratio = batch_expansions / (
                        len(world_batch_print_df) * expansion_budget
                    )

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        fb_exp_ratio = (
                            world_batch_print_df["FExp"].sum()
                            / world_batch_print_df["BExp"].sum()
                        )
                        fb_g_ratio = (
                            world_batch_print_df["Fg"].sum()
                            / world_batch_print_df["Bg"].sum()
                        )

                    print(
                        f"{'Solved':23s}: {num_problems_solved_this_batch}/{num_problems_this_batch}"
                    )
                    print(
                        f"{'Total expansion ratio':23s}: {batch_expansions_ratio:.3f}"
                    )
                    print(f"{'F/B expansion ratio':23s}: {fb_exp_ratio:.3f}")
                    print(f"{'F/B g-cost ratio':23s}: {fb_g_ratio:.3f}\n")

                    writer.add_scalar(
                        f"expansions_vs_batch", batch_expansions_ratio, batches_seen
                    )
                    writer.add_scalar(
                        f"fb_expansions_ratio_vs_batch", fb_exp_ratio, batches_seen
                    )
                    writer.add_scalar(f"fb_g_ratio_vs_batch", fb_g_ratio, batches_seen)

                    total_num_expanded += world_batch_print_df["Exp"].sum()
                    writer.add_scalar(
                        "cum_unique_solved_vs_expanded",
                        len(solved_problems),
                        total_num_expanded,
                    )
                    # if batches_seen % param_log_interval == 0:
                    #     log_params(writer, model, batches_seen)
                    # writer.add_scalar(f"cum_unique_solved_vs_batch", len(solved_problems), batches_seen)
                    sys.stdout.flush()

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

                        local_batch_opt_results[0] = f_avg_action_nll
                        local_batch_opt_results[1] = f_acc
                        local_batch_opt_results[2] = 1

                        if bidirectional:
                            (
                                b_loss,
                                b_avg_action_nll,
                                b_acc,
                            ) = loss_fn(b_trajs, model)

                            local_batch_opt_results[3] = b_avg_action_nll
                            local_batch_opt_results[4] = b_acc

                            loss = f_loss + b_loss
                        else:
                            loss = f_loss

                        loss.backward()
                    else:
                        local_batch_opt_results[:] = 0

                    if is_distributed:
                        dist.all_reduce(local_batch_opt_results, op=dist.ReduceOp.SUM)
                        num_procs_found_solution = int(
                            local_batch_opt_results[2].item()
                        )
                        if num_procs_found_solution > 0:
                            sync_grads(model, num_procs_found_solution)
                    else:
                        num_procs_found_solution = int(
                            local_batch_opt_results[2].item()
                        )

                    optimizer.step()

                    if num_procs_found_solution > 0:
                        if rank == 0:
                            if grad_step == 1 or grad_step == grad_steps:
                                if grad_step == 1:
                                    print(opt_result_header)

                                f_loss = (
                                    local_batch_opt_results[0].item()
                                    / num_procs_found_solution
                                )
                                f_acc = (
                                    local_batch_opt_results[1].item()
                                    / num_procs_found_solution
                                )
                                b_loss = (
                                    local_batch_opt_results[3].item()
                                    / num_procs_found_solution
                                )
                                b_acc = (
                                    local_batch_opt_results[4].item()
                                    / num_procs_found_solution
                                )
                                if bidirectional:
                                    print(
                                        f"{opt_step:7}  {f_loss:5.3f}  {f_acc:5.3f}    {b_loss:5.3f}  {b_acc:5.3f}"
                                    )
                                else:
                                    print(f"{opt_step:7}  {f_loss:5.3f}  {f_acc:5.3f}")
                                if grad_step == grad_steps:
                                    # fmt: off
                                    writer.add_scalar( f"loss_vs_opt_pass/forward", f_loss, opt_passes,)
                                    writer.add_scalar( f"acc_vs_opt_pass/forward", f_acc, opt_passes,)
                                    if bidirectional:
                                        writer.add_scalar( f"loss_vs_opt_pass/backward", b_loss, opt_passes,)
                                        writer.add_scalar( f"acc_vs_opt_pass/backward", b_acc, opt_passes,)
                                    # fmt:on
                        opt_step += 1
                if num_procs_found_solution > 0:
                    opt_passes += 1

                world_epoch_f_loss[batch_idx] = f_loss
                world_epoch_f_acc[batch_idx] = f_acc
                if bidirectional:
                    world_epoch_b_loss[batch_idx] = b_loss
                    world_epoch_b_acc[batch_idx] = b_acc

            if rank == 0:
                epoch_expansions = world_results_df["Exp"].sum()
                epoch_expansions_ratio = epoch_expansions / max_epoch_expansions
                epoch_solved_ratio = num_problems_solved_this_epoch / world_num_problems
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    epoch_fb_exp_ratio = (
                        world_results_df["FExp"].sum() / world_results_df["BExp"].sum()
                    )
                    epoch_fb_g_ratio_ratio = (
                        world_results_df["Fg"].sum() / world_results_df["Bg"].sum()
                    )
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
                    f"END | STAGE {train_loader.stage} EPOCH {stage_epoch} | TOTAL EPOCH {epoch}"
                )
                print(
                    "----------------------------------------------------------------------------"
                )
                print(
                    f"{'Solved':20s}: {num_problems_solved_this_epoch}/{world_num_problems} {epoch_solved_ratio}\n"
                    f"{'Expansions':20s}: {int(epoch_expansions)}/{max_epoch_expansions}  {epoch_expansions_ratio:5.3f}\n"
                    f"{'FB Exp Ratio':20s}: {epoch_fb_exp_ratio:5.3f}\n"
                    f"{'FB G-Cost Ratio':20s}: {epoch_fb_g_ratio_ratio:5.3f}\n"
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

                # fmt: off
                # writer.add_scalar("budget_vs_epoch", budget, epoch)
                # writer.add_scalar(f"budget_{budget}/solved_vs_epoch", epoch_solved, epoch)
                writer.add_scalar(f"solved_vs_epoch", epoch_solved_ratio, epoch)
                writer.add_scalar("cum_unique_solved_vs_epoch", len(solved_problems), epoch)

                writer.add_scalar(f"loss_vs_epoch/forward", epoch_f_loss, epoch)
                writer.add_scalar(f"acc_vs_epoch/forward", epoch_f_acc, epoch)
                if bidirectional:
                    writer.add_scalar(f"loss_vs_epoch/backward", epoch_b_loss, epoch)
                    writer.add_scalar(f"acc_vs_epoch/backward", epoch_b_acc, epoch)

                # writer.add_scalar(f"expansions_vs_epoch", expansions, epoch)
                writer.add_scalar(f"expansions_ratio_vs_epoch", epoch_expansions_ratio, epoch)
                writer.add_scalar(f"fb_expansions_ratio_vs_epoch", epoch_fb_exp_ratio, epoch)
                writer.add_scalar(f"fb_g_ratio_vs_epoch", epoch_fb_g_ratio_ratio, epoch)

                world_results_df.to_pickle(epoch_logdir / "train.pkl")
                # fmt: on
                sys.stdout.flush()

            if valid_loader and epoch >= epoch_begin_validate:
                if rank == 0:
                    print("VALIDATION")
                if is_distributed:
                    dist.barrier()
                valid_results = test(
                    rank,
                    agent,
                    valid_loader,
                    writer,
                    world_size,
                    expansion_budget,
                    time_budget,
                    increase_budget=False,
                    print_results=False,
                    validate=True,
                    epoch=epoch,
                )

                if rank == 0:
                    (
                        valid_solved,
                        valid_total_expanded,
                        valid_fb_exp_ratio,
                        valid_fb_g_ratio,
                    ) = valid_results
                    valid_expansions_ratio = valid_total_expanded / max_valid_expanded
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
                    writer.add_scalar(f"valid_solved_vs_epoch", valid_solve_rate, epoch)
                    writer.add_scalar(
                        f"valid_expansions_ratio_vs_epoch",
                        valid_expansions_ratio,
                        epoch,
                    )
                    writer.add_scalar(
                        f"valid_fb_exp_ratio_vs_epoch", valid_fb_exp_ratio, epoch
                    )
                    writer.add_scalar(f"valid_fb_g_ratio", valid_fb_g_ratio, epoch)
                    # writer.add_scalar( f"valid_expanded_vs_epoch", valid_total_expanded, epoch)

                    agent.save_model("latest", log=False)
                    if valid_total_expanded <= best_valid_total_expanded:
                        best_valid_total_expanded = valid_total_expanded
                        print("Saving best model by expansions")
                        writer.add_text(
                            "best_model_expansions",
                            f"epoch: {epoch}, solve rate: {valid_solve_rate}, expansion ratio: {valid_expansions_ratio}",
                        )
                        agent.save_model("best_expanded", log=False)

                    if valid_solved >= best_valid_solved:
                        best_valid_solved = valid_solved
                        print("Saving best model by solved")
                        writer.add_text(
                            "best_model_solved",
                            f"epoch: {epoch}, solve rate: {valid_solve_rate}, expansion ratio: {valid_expansions_ratio}",
                        )
                        agent.save_model("best_solved", log=False)

                    if isinstance(model, to.jit.ScriptModule):
                        ext = ".ts"
                    else:
                        ext = ".pt"
                    copyfile(
                        logdir / f"model_latest{ext}",
                        epoch_logdir / f"model_latest{ext}",
                    )
                    copyfile(
                        logdir / f"model_best_solved{ext}",
                        epoch_logdir / f"model_best_solved{ext}",
                    )
                    copyfile(
                        logdir / f"model_best_expanded{ext}",
                        epoch_logdir / f"model_best_expanded{ext}",
                    )

                if is_distributed:
                    dist.barrier()

            epoch += 1

    # all epochs completed
    if rank == 0:
        print("END TRAINING")


def log_params(writer, model, batches_seen):
    for (
        param_name,
        param,
    ) in model.named_parameters():
        writer.add_histogram(
            f"param_vs_batch/{param_name}",
            param.data,
            batches_seen,
            bins=512,
        )


def log_grad_norm(parameters, name, writer, opt_step):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    writer.add_scalar(f"total_grad_norm/{name}", total_norm, opt_step)


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
