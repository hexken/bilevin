import csv
from pathlib import Path
import pickle
import sys
from timeit import default_timer as timer
import warnings

import numpy as np
import pandas as pd
from tabulate import tabulate
from test import test
import torch as to
import torch.distributed as dist

from loaders import ProblemLoader
from search.agent import Agent
from search.utils import (
    int_columns,
    print_model_train_summary,
    print_search_summary,
    search_result_header,
)


def train(
    args,
    rank: int,
    agent: Agent,
    train_loader: ProblemLoader,
    valid_loader: ProblemLoader,
    seed: int,
):
    logdir = args.logdir
    world_size = args.world_size
    time_budget = args.time_budget
    expansion_budget = args.expansion_budget
    grad_steps = args.grad_steps

    if rank == 0:
        best_csv = (logdir / "best_models.csv").open("w", newline="")
        best_writer = csv.DictWriter(best_csv, ["epoch", "solve_rate", "exp_ratio"])
        best_writer.writeheader()

    bidirectional = agent.bidirectional
    model = agent.model
    optimizer = agent.optimizer
    loss_fn = agent.loss_fn

    for param in model.parameters():
        if not param.grad:
            param.grad = to.zeros_like(param)

    local_opt_results = to.zeros(5, dtype=to.float64)
    local_search_results = to.zeros(len(search_result_header), dtype=to.float64)

    world_search_results = [
        to.zeros(len(search_result_header), dtype=to.float64)
        for _ in range(args.world_size)
    ]

    batches_seen = 0
    batch_size = args.world_size

    num_valid_problems = len(valid_loader)
    max_valid_expanded = num_valid_problems * expansion_budget
    best_valid_solved = -1
    best_valid_total_expanded = max_valid_expanded + 1

    stage_ids = []
    stage_times = []
    stage_fexps = []
    stage_bexps = []
    stage_flens = []
    stage_blens = []
    stage_fpnlls = []
    stage_bpnlls = []
    stage_fnlls = []
    stage_bnlls = []

    stage_flosses = []
    stage_faccs = []
    stage_blosses = []
    stage_baccs = []
    batches = []

    # for running avg of stage solve rate
    stage_avg = 0
    stage_problems_seen = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = timer()

        epoch_search_dfs = []
        epoch_model_train_dfs = []

        if rank == 0:
            print(
                "============================================================================"
            )
            print(f"START EPOCH {epoch}")

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

        old_stage = -1
        for problem in train_loader:
            if old_stage != train_loader.stage:
                old_stage = train_loader.stage
                stage_start_time = timer()
                print(
                    "----------------------------------------------------------------------------"
                )
                print(f"START STAGE {old_stage}")
                print(
                    "----------------------------------------------------------------------------"
                )

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

            local_search_results[0] = problem.id
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

            batch_print_df = pd.DataFrame(
                batch_results_arr, columns=search_result_header
            )
            for col in int_columns:
                batch_print_df[col] = batch_print_df[col].astype(int)
            batch_print_df = batch_print_df.sort_values("Exp")

            if rank == 0:
                print(f"\n\nBatch {batches_seen}")

            if rank == 0:
                print(
                    tabulate(
                        batch_print_df,
                        headers="keys",
                        tablefmt="psql",
                        showindex=False,
                        # floatfmt=".2f"
                        # intfmt="",
                    )
                )

            for i in range(batch_size):
                stage_ids.append(batch_results_arr[i, 0])
                stage_times.append(batch_results_arr[i, 1])
                stage_fexps.append(batch_results_arr[i, 3])
                stage_bexps.append(batch_results_arr[i, 4])
                stage_flens.append(batch_results_arr[i, 6])
                stage_blens.append(batch_results_arr[i, 7])
                stage_fpnlls.append(batch_results_arr[i, 8])
                stage_bpnlls.append(batch_results_arr[i, 9])
                stage_fnlls.append(batch_results_arr[i, 10])
                stage_bnlls.append(batch_results_arr[i, 11])

            # perform grad steps
            dist.all_reduce(local_opt_results, op=dist.ReduceOp.SUM)
            num_procs_found_solution = int(local_opt_results[2].item())

            batches_seen += 1

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

                    if grad_step == grad_steps:
                        batch_floss = (
                            local_opt_results[0].item() / num_procs_found_solution
                        )
                        batch_facc = (
                            local_opt_results[1].item() / num_procs_found_solution
                        )
                        batch_bloss = (
                            local_opt_results[3].item() / num_procs_found_solution
                        )
                        batch_bacc = (
                            local_opt_results[4].item() / num_procs_found_solution
                        )

                        stage_flosses.append(batch_floss)
                        stage_faccs.append(batch_facc)
                        if bidirectional:
                            stage_blosses.append(batch_bloss)
                            stage_baccs.append(batch_bacc)
                        batches.append(batches_seen)
                        print(f"floss: {batch_floss:.3f}")
                        print(f"facc: {batch_facc:.3f}")
                        if bidirectional:
                            print(f"\nbloss: {batch_bloss:.3f}")
                            print(f"bacc: {batch_bacc:.3f}")

            num_problems_solved_this_batch = len(
                batch_print_df[batch_print_df["Len"] > 0]["Len"]
            )

            batch_avg = num_problems_solved_this_batch / batch_size
            stage_avg += (batch_avg - stage_avg) / (batches_seen)

            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore", category=RuntimeWarning)

            if (
                train_loader.manual_advance
                and stage_problems_seen >= args.min_samples_per_stage
                and stage_avg >= args.min_stage_solve_ratio
            ):
                train_loader.goto_next_stage = True

            if train_loader.goto_next_stage:
                if bidirectional:
                    stage_search_df = {
                        "id": pd.Series(stage_ids, dtype=np.uint32),
                        "time": pd.Series(stage_times, dtype=np.float32),
                        "fexp": pd.Series(stage_fexps, dtype=np.uint16),
                        "bexp": pd.Series(stage_bexps, dtype=np.uint16),
                        "flen": pd.Series(stage_flens, dtype=np.uint16),
                        "blen": pd.Series(stage_blens, dtype=np.uint16),
                        "fpnll": pd.Series(stage_fpnlls, dtype=np.float32),
                        "bpnll": pd.Series(stage_bpnlls, dtype=np.float32),
                        "fnll": pd.Series(stage_fnlls, dtype=np.float32),
                        "bnll": pd.Series(stage_bnlls, dtype=np.float32),
                        "stage": pd.Series(
                            [old_stage for _ in range(stage_problems_seen)],
                            dtype=np.uint8,
                        ),
                    }
                    stage_model_train_df = {
                        "floss": pd.Series(stage_flosses, dtype=np.float32),
                        "bloss": pd.Series(stage_blosses, dtype=np.float32),
                        "facc": pd.Series(stage_faccs, dtype=np.float32),
                        "bacc": pd.Series(stage_baccs, dtype=np.float32),
                        "batch": pd.Series(batches, dtype=np.uint32),
                    }
                else:
                    stage_search_df = {
                        "id": pd.Series(stage_ids, dtype=np.uint32),
                        "time": pd.Series(stage_times, dtype=np.float32),
                        "fexp": pd.Series(stage_fexps, dtype=np.uint16),
                        "flen": pd.Series(stage_flens, dtype=np.uint16),
                        "fpnll": pd.Series(stage_fpnlls, dtype=np.float32),
                        "fnll": pd.Series(stage_fnlls, dtype=np.float32),
                        "stage": pd.Series(
                            [old_stage for _ in range(stage_problems_seen)],
                            dtype=np.uint8,
                        ),
                    }
                    stage_model_train_df = {
                        "floss": pd.Series(stage_flosses, dtype=np.float32),
                        "facc": pd.Series(stage_faccs, dtype=np.float32),
                        "batch": pd.Series(batches, dtype=np.uint32),
                    }

                epoch_search_dfs.append(stage_search_df)
                epoch_model_train_dfs.append(stage_model_train_df)

                stage_times = []
                stage_fexps = []
                stage_bexps = []
                stage_flens = []
                stage_blens = []
                stage_fpnlls = []
                stage_bpnlls = []
                stage_fnlls = []
                stage_bnlls = []

                stage_flosses = []
                stage_faccs = []
                stage_blosses = []
                stage_baccs = []
                batches = []

                print(
                    "----------------------------------------------------------------------------"
                )
                print(f"END STAGE {old_stage}")
                print_search_summary(
                    stage_search_df,
                    bidirectional,
                )
                print_model_train_summary(
                    stage_model_train_df,
                    bidirectional,
                )
                print(f"Time: {timer() - stage_start_time:.2f}")
                print(
                    "----------------------------------------------------------------------------"
                )
            # BATCH END

        # process end of epoch
        if rank == 0:
            print(
                "============================================================================"
            )
            print(f"END EPOCH {epoch}")
            epoch_search_df = pd.concat(epoch_search_dfs, ignore_index=True)
            epoch_model_train_df = pd.concat(epoch_model_train_dfs, ignore_index=True)
            print_search_summary(
                epoch_search_df,
                bidirectional,
            )
            with (logdir / f"search_train_{epoch}.pkl").open("wb") as f:
                pickle.dump(epoch_search_df, f)

            epoch_model_train_df = pd.concat(epoch_model_train_dfs, ignore_index=True)
            print_model_train_summary(
                epoch_model_train_df,
                bidirectional,
            )
            with (logdir / f"model_train_{epoch}.pkl").open("wb") as f:
                pickle.dump(epoch_model_train_df, f)
            print(f"\nTime: {timer() - epoch_start_time:.2f}")
            print(
                "============================================================================"
            )

        if epoch >= args.epoch_begin_validate and epoch % args.validate_every == 0:
            if rank == 0:
                print("VALIDATION")

            dist.barrier()

            valid_results = test(
                rank,
                logdir,
                agent,
                valid_loader,
                world_size,
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
                ) = valid_results

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
