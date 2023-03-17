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

import math
from math import isclose
from pathlib import Path
import sys
import time
from typing import Callable, Type, Union
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from tabulate import tabulate
from test import test
import torch as to
import torch.distributed as dist
from torch.multiprocessing import Queue
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard.writer import SummaryWriter
import tqdm

from domains.domain import Problem
from loaders import CurriculumLoader, ProblemsBatchLoader
from search import MergedTrajectory
from search.agent import Agent


def train(
    rank: int,
    agent: Agent,
    model: Union[to.nn.Module, tuple[to.nn.Module, to.nn.Module]],
    model_save_path: Path,
    loss_fn: Callable,
    optimizer_cons: Type[to.optim.Optimizer],
    optimizer_params: dict,
    train_loader: CurriculumLoader,
    writer: SummaryWriter,
    world_size: int,
    update_levin_costs: bool,
    budget: int,
    seed: int,
    grad_steps: int = 10,
    epochs_reduce_lr: int = 5,
    shuffle_trajectory: bool = False,
    valid_loader: Optional[ProblemsBatchLoader] = None,
    results_queue: Optional[Queue] = None,
):
    is_distributed = world_size > 1

    param_log_interval = 20

    search_result_header = [
        "ProblemId",
        "SolutionLength",
        "NumExpanded",
        "NumGenerated",
        "Time",
    ]

    opt_result_header = f"OptStep   Loss    Acc"

    bidirectional = agent.bidirectional
    if bidirectional:
        assert isinstance(model, tuple)
        f_model, b_model = model
        f_model_save_path = model_save_path / "forward_best.pt"
        b_model_save_path = model_save_path / "backward_best.pt"

        forward_optimizer = optimizer_cons(f_model.parameters(), **optimizer_params)
        backward_optimizer = optimizer_cons(b_model.parameters(), **optimizer_params)

        for param in b_model.parameters():
            if not param.grad:
                param.grad = to.zeros_like(param)
    else:
        assert isinstance(model, to.nn.Module)
        f_model = model
        f_model_save_path = model_save_path / "forward_best.pt"
        forward_optimizer = optimizer_cons(f_model.parameters(), **optimizer_params)

    for param in f_model.parameters():
        if not param.grad:
            param.grad = to.zeros_like(param)

    if rank == 0:
        log_params(writer, f_model, "forward", 0)
        if bidirectional:
            log_params(writer, b_model, "backward", 0)

    local_batch_opt_results = to.zeros(3, dtype=to.float64)

    local_batch_size = train_loader.batch_size
    local_batch_search_results = to.zeros(local_batch_size, 5, dtype=to.int64)
    world_batch_search_results = [
        to.zeros((local_batch_size, 5), dtype=to.int64) for _ in range(world_size)
    ]

    batches_seen = 0
    solved_problems = set()
    total_num_expanded = 0
    forward_opt_steps = 0
    backward_opt_steps = 0

    best_valid_solve_rate = 0.0
    max_valid_expanded = 0 if not valid_loader else (len(valid_loader.all_ids) * budget)
    best_valid_total_expanded = max_valid_expanded

    epoch = 1
    for batch_loader in train_loader:
        print(f"{train_loader.stage} NEW BATCH LOADER")
        world_num_problems = len(batch_loader.all_ids)

        world_batches_this_difficulty = math.ceil(
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
        world_results_df["Time"] = world_results_df["Time"].astype(float, copy=False)
        world_results_df["ProblemId"] = batch_loader.all_ids
        world_results_df.set_index("ProblemId", inplace=True)

        world_epoch_f_loss = np.zeros(world_batches_this_difficulty)
        world_epoch_f_acc = np.zeros(world_batches_this_difficulty)
        world_epoch_b_loss = np.zeros(world_batches_this_difficulty)
        world_epoch_b_acc = np.zeros(world_batches_this_difficulty)

        for stage_epoch in range(1, batch_loader.epochs + 1):
            num_new_problems_solved_this_epoch = 0
            num_problems_solved_this_epoch = 0

            if rank == 0:
                print(
                    "============================================================================"
                )
                print(
                    f"START | STAGE {train_loader.stage} EPOCH {stage_epoch} | TOTAL EPOCH {epoch}"
                )
                print(
                    "============================================================================\n"
                )

            if rank == 0:
                problems_loader = tqdm.tqdm(
                    batch_loader, total=world_batches_this_difficulty
                )
            else:
                problems_loader = batch_loader

            for batch_idx, local_batch_problems in enumerate(problems_loader):
                # if rank != 0:
                #     print(f"rank {rank} bs {problems_loader.batches_served}")
                # else:
                #     print(f"rank {rank} bs {problems_loader.iterable.batches_served}")

                batches_seen += 1
                local_batch_search_results[
                    :
                ] = 0  # since final batch might contain <= local_batch_size problems

                if rank == 0:
                    print(f"\n\nBatch {batches_seen}")

                to.set_grad_enabled(False)

                forward_trajs = []
                f_model.eval()
                backward_trajs = []
                if bidirectional:
                    b_model.eval()  # type:ignore

                num_problems_solved_this_batch = 0
                for i, problem in enumerate(local_batch_problems):
                    start_time = time.time()
                    (
                        solution_length,
                        num_expanded,
                        num_generated,
                        traj,
                    ) = agent.search(
                        problem,
                        model,
                        budget,
                        update_levin_costs,
                        train=True,
                    )
                    end_time = time.time()
                    if bidirectional:
                        problem.domain.reset()

                    # print(problem.id)
                    local_batch_search_results[i, 0] = problem.id_idx
                    local_batch_search_results[i, 1] = solution_length
                    local_batch_search_results[i, 2] = num_expanded
                    local_batch_search_results[i, 3] = num_generated
                    local_batch_search_results[i, 4] = int(
                        (end_time - start_time) * 1000
                    )

                    if traj:
                        forward_trajs.append(traj[0])
                        if bidirectional:
                            backward_trajs.append(traj[1])

                if is_distributed:
                    dist.all_gather(
                        world_batch_search_results, local_batch_search_results
                    )
                    world_batch_results_t = to.cat(world_batch_search_results, dim=0)
                else:
                    world_batch_results_t = local_batch_search_results

                world_batch_results_arr = world_batch_results_t.numpy()
                # hacky way to filter out results from partial batches
                world_batch_results_arr = world_batch_results_arr[
                    world_batch_results_arr[:, 2] > 0
                ]

                # print(world_batch_results_arr[:,0])
                # print(batch_loader.all_ids)
                world_batch_ids = np.array(
                    [batch_loader.all_ids[i] for i in world_batch_results_arr[:, 0]],
                    dtype=np.unicode_,
                )
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
                    total_num_expanded += world_batch_results_df["NumExpanded"].sum()

                    writer.add_scalar(
                        "cum_unique_solved_vs_expanded",
                        len(solved_problems),
                        total_num_expanded,
                    )

                def fit_model(
                    model: to.nn.Module,
                    optimizer: to.optim.Optimizer,
                    trajs: list,
                    opt_step: int,
                    name: str,
                ):
                    if rank == 0:
                        print(f"{name}:")
                        print(opt_result_header)

                    merged_traj = MergedTrajectory(trajs, shuffle_trajectory)
                    to.set_grad_enabled(True)
                    model.train()

                    num_procs_found_solution = 0
                    loss = -1
                    acc = -1
                    for _ in range(grad_steps):
                        optimizer.zero_grad()
                        if trajs:
                            loss, avg_action_nll, logits = loss_fn(merged_traj, model)
                            loss.backward()

                            acc = (
                                logits.argmax(dim=1) == merged_traj.actions
                            ).sum().item() / len(logits)

                            local_batch_opt_results[0] = avg_action_nll
                            local_batch_opt_results[1] = acc
                            local_batch_opt_results[2] = 1
                        else:
                            local_batch_opt_results[:] = 0

                        if is_distributed:
                            dist.all_reduce(
                                local_batch_opt_results, op=dist.ReduceOp.SUM
                            )
                            num_procs_found_solution = int(
                                local_batch_opt_results[2].item()
                            )
                            if num_procs_found_solution > 0:
                                sync_grads(model, num_procs_found_solution)

                        num_procs_found_solution = int(
                            local_batch_opt_results[2].item()
                        )

                        # todo grad clipping? for now inspect norms
                        if trajs and rank == 0:
                            total_norm = 0
                            for p in model.parameters():
                                param_norm = p.grad.detach().data.norm(2)
                                total_norm += param_norm.item() ** 2
                            total_norm = total_norm**0.5
                            writer.add_scalar(
                                f"total_grad_norm/{name}", total_norm, opt_step
                            )

                        optimizer.step()

                        if num_procs_found_solution > 0:
                            opt_step += 1
                            if rank == 0:
                                opt_passes = opt_step // grad_steps
                                step_within_opt_pass = opt_step % grad_steps
                                if (
                                    step_within_opt_pass == 1
                                    or step_within_opt_pass == 0
                                ):
                                    loss = (
                                        local_batch_opt_results[0].item()
                                        / num_procs_found_solution
                                    )
                                    acc = (
                                        local_batch_opt_results[1].item()
                                        / num_procs_found_solution
                                    )
                                    print(f"{opt_step:7}  {loss:5.3f}  {acc:5.3f}")
                                    if step_within_opt_pass == 0:
                                        # fmt: off
                                        writer.add_scalar( f"loss_vs_opt_pass/{name}", loss, opt_passes,)
                                        writer.add_scalar( f"acc_vs_opt_pass/{name}", acc, opt_passes,)
                                        # fmt:on

                    if rank == 0 and num_procs_found_solution > 0:
                        print("")
                    return opt_step, loss, acc

                forward_opt_steps, f_loss, f_acc = fit_model(
                    f_model,
                    forward_optimizer,
                    forward_trajs,
                    forward_opt_steps,
                    name="forward",
                )
                world_epoch_f_loss[batch_idx] = f_loss
                world_epoch_f_acc[batch_idx] = f_acc

                if bidirectional:
                    backward_opt_steps, b_loss, b_acc = fit_model(
                        b_model,  # type:ignore
                        backward_optimizer,  # type:ignore
                        backward_trajs,  # type:ignore
                        backward_opt_steps,  # type:ignore
                        name="backward",
                    )
                    world_epoch_b_loss[batch_idx] = b_loss
                    world_epoch_b_acc[batch_idx] = b_acc

                if rank == 0:
                    if batches_seen % param_log_interval == 0:
                        log_params(writer, f_model, "forward", batches_seen)
                        if bidirectional:
                            log_params(writer, b_model, "backward", batches_seen)

                    batch_avg = num_problems_solved_this_batch / num_problems_this_batch
                    # fmt: off
                    writer.add_scalar(f"solved_vs_batch", batch_avg, batches_seen)
                    writer.add_scalar(f"cum_unique_solved_vs_batch", len(solved_problems), batches_seen)
                    # fmt: on
                    sys.stdout.flush()

            if rank == 0:
                expansions = world_results_df["NumExpanded"].sum()
                expansions_ratio = expansions / (world_num_problems * budget)
                epoch_solved = num_problems_solved_this_epoch / world_num_problems
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
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
                    f"SOLVED {num_problems_solved_this_epoch}/{world_num_problems}  {num_problems_solved_this_epoch / world_num_problems}\n"
                    f"EXPANSIONS {expansions}/{world_num_problems * budget}  {expansions_ratio}  \n"
                    f"Average forward loss: {epoch_f_loss:5.3f}, acc: {epoch_f_acc:5.3f}"
                )
                if bidirectional:
                    print(
                        f"Average backward loss: {epoch_b_loss:5.3f}, acc: {epoch_b_acc:5.3f}"
                    )
                print(
                    "============================================================================"
                )

                # fmt: off
                # writer.add_scalar("budget_vs_epoch", budget, epoch)
                # writer.add_scalar(f"budget_{budget}/solved_vs_epoch", epoch_solved, epoch)
                writer.add_scalar(f"solved_vs_epoch", epoch_solved, epoch)
                writer.add_scalar("cum_unique_solved_vs_epoch", len(solved_problems), epoch)

                writer.add_scalar(f"loss_vs_epoch/forward", epoch_f_loss, epoch)
                writer.add_scalar(f"acc_vs_epoch/forward", epoch_f_acc, epoch)
                if bidirectional:
                    writer.add_scalar(f"loss_vs_epoch/backward", epoch_b_loss, epoch)
                    writer.add_scalar(f"acc_vs_epoch/backward", epoch_b_acc, epoch)

                world_results_df.to_csv(f"{writer.log_dir}/epoch_{epoch}.csv")
                writer.add_scalar(f"expansions_vs_epoch/forward", expansions_ratio, epoch)
                writer.add_scalar(f"expansions_ratio_vs_epoch/forward", expansions_ratio, epoch)
                # fmt: on
                sys.stdout.flush()

            if valid_loader:
                # print(f"rank {rank}")
                if rank == 0:
                    print("VALIDATION")
                if is_distributed:
                    dist.barrier()
                valid_results = test(
                    rank,
                    agent,
                    model,
                    valid_loader,
                    writer,
                    world_size,
                    update_levin_costs,
                    budget,
                    results_queue,
                    increase_budget=False,
                    validate=True,
                    epoch=epoch,
                )

                if rank == 0:
                    valid_solved, valid_total_expanded = valid_results
                    valid_expansions_rate = valid_total_expanded / max_valid_expanded
                    valid_solve_rate = valid_solved / len(valid_loader.all_ids)
                    print(
                        f"SOLVED:  {valid_solved}/{len(valid_loader.all_ids)}  {valid_solve_rate}"
                    )
                    print(
                        f"EXPANSIONS: {valid_total_expanded}/{max_valid_expanded} {valid_expansions_rate}"
                    )
                    # writer.add_scalar(f"budget_{budget}/valid_solve_rate", valid_solve_rate, epoch)
                    writer.add_scalar(f"valid_solved_vs_epoch", valid_solve_rate, epoch)
                    writer.add_scalar(
                        f"expansions_ratio_vs_epoch/forward",
                        valid_expansions_rate,
                        epoch,
                    )
                    writer.add_scalar(
                        f"valid_expanded_vs_epoch", valid_total_expanded, epoch
                    )

                    if valid_solve_rate > best_valid_solve_rate or (
                        isclose(valid_solve_rate, best_valid_solve_rate)
                        and valid_total_expanded < best_valid_total_expanded
                    ):
                        best_valid_solve_rate = valid_solve_rate
                        best_valid_total_expanded = valid_total_expanded
                        print("Saving best model...")
                        writer.add_text("best_model", f"epoch {epoch}")
                        to.save(
                            f_model.state_dict(),
                            f_model_save_path.with_name("forward_best.pt"),
                        )
                        if bidirectional:
                            to.save(
                                b_model.state_dict(),
                                b_model_save_path.with_name("backward_best.pt"),
                            )

                if is_distributed:
                    dist.barrier()

            # todo clean this up
            if epoch == epochs_reduce_lr:
                new_lr = forward_optimizer.param_groups[0]["lr"] * 0.1
                if rank == 0:
                    print(
                        f"Reducing learning rate from {forward_optimizer.param_groups[0]['lr']} to {new_lr}"
                    )
                forward_optimizer.param_groups[0]["lr"] = new_lr
                if bidirectional:
                    backward_optimizer.param_groups[0]["lr"] = new_lr

            epoch += 1

    # all epochs completed
    if rank == 0:
        to.save(f_model.state_dict(), f_model_save_path)
        if bidirectional:
            to.save(b_model.state_dict(), b_model_save_path)
        print("END TRAINING")


def log_params(writer, model, name, batches_seen):
    for (
        param_name,
        param,
    ) in model.named_parameters():
        writer.add_histogram(
            f"param_vs_batch/{name}/{param_name}",
            param.data,
            batches_seen,
            bins=512,
        )


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
