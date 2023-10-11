import pickle
import sys
from timeit import default_timer as timer

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
    batch_size = args.world_size
    time_budget = args.time_budget
    expansion_budget = args.expansion_budget
    grad_steps = args.grad_steps

    if rank == 0:
        best_models_log = (logdir / "best_models.txt").open("w")

    bidirectional = agent.bidirectional
    model = agent.model
    optimizer = agent.optimizer
    loss_fn = agent.loss_fn

    for param in model.parameters():
        if not param.grad:
            param.grad = to.zeros_like(param)

    solved_flag = to.zeros(1, dtype=to.uint8)
    local_opt_results = to.zeros(4, dtype=to.float64)

    world_search_results = [
        to.zeros(len(search_result_header), dtype=to.float64) for _ in range(batch_size)
    ]

    total_batches_seen = 0

    num_valid_problems = len(valid_loader)
    max_valid_expanded = num_valid_problems * expansion_budget
    best_valid_solved = -1
    best_valid_total_expanded = max_valid_expanded + 1

    # to suppress possible unbound warnings
    ids = []
    times = []
    lens = []
    fexps = []
    bexps = []
    fgs = []
    bgs = []
    fpnlls = []
    bpnlls = []
    fnlls = []
    bnlls = []

    flosses = []
    faccs = []
    blosses = []
    baccs = []

    stage_avg = 0
    stage_batches_seen = 0
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
                "============================================================================"
            )

        old_stage = train_loader.stage
        for problem in train_loader:
            if old_stage != train_loader.stage:
                # if rank == 0:
                #     for p in train_loader.stage_problems:
                #         print(p.id)
                stage_start_time = timer()
                old_stage = train_loader.stage

                ids = []
                times = []
                lens = []
                fexps = []
                bexps = []
                fgs = []
                bgs = []
                fpnlls = []
                bpnlls = []
                fnlls = []
                bnlls = []

                flosses = []
                faccs = []
                blosses = []
                baccs = []

                stage_avg = 0
                stage_batches_seen = 0
                stage_problems_seen = 0

                if rank == 0:
                    print(
                        "----------------------------------------------------------------------------"
                    )
                    print(f"START STAGE {old_stage}")
                    print(
                        "----------------------------------------------------------------------------"
                    )
            model.eval()
            to.set_grad_enabled(False)

            start_time = timer()
            (
                n_forw_expanded,
                n_backw_expanded,
                traj,
            ) = agent.search(
                problem,
                expansion_budget,
                time_budget=time_budget,
            )
            end_time = timer()
            solution_length = np.nan if not traj else traj[0].cost

            if bidirectional:
                problem.domain.reset()

            local_search_results = to.zeros(len(search_result_header), dtype=to.float64)
            local_search_results[0] = problem.id
            local_search_results[1] = end_time - start_time
            local_search_results[2] = n_forw_expanded + n_backw_expanded
            local_search_results[3] = n_forw_expanded
            local_search_results[4] = n_backw_expanded
            local_search_results[5] = solution_length

            if traj:
                solved_flag[0] = 1
                f_traj, b_traj = traj
                local_search_results[6] = f_traj.partial_g_cost
                local_search_results[8] = -1 * f_traj.partial_log_prob
                local_search_results[10] = -1 * f_traj.log_prob
                if b_traj:
                    local_search_results[7] = b_traj.partial_g_cost
                    local_search_results[9] = -1 * b_traj.partial_log_prob
                    local_search_results[11] = -1 * b_traj.log_prob

            else:
                local_search_results[6:] = np.nan
                f_traj = b_traj = None
                solved_flag[0] = 0

            dist.all_gather(world_search_results, local_search_results)
            dist.all_reduce(solved_flag, op=dist.ReduceOp.SUM)
            num_procs_found_solution = solved_flag[0].item()

            world_batch_results_t = to.stack(world_search_results, dim=0)
            batch_results_arr = world_batch_results_t.numpy()
            batch_print_df = pd.DataFrame(
                batch_results_arr, columns=search_result_header
            )
            for col in int_columns:
                batch_print_df[col] = batch_print_df[col].astype(int)
            batch_print_df = batch_print_df.sort_values("exp")

            if rank == 0:
                print(f"\nBatch {total_batches_seen}")

            if rank == 0:
                print(
                    tabulate(
                        batch_print_df,
                        headers="keys",
                        tablefmt="psql",
                        showindex=False,
                    )
                )

            for i in range(batch_size):
                ids.append(batch_results_arr[i, 0])
                times.append(batch_results_arr[i, 1])
                lens.append(batch_results_arr[i, 5])
                fexps.append(batch_results_arr[i, 3])
                bexps.append(batch_results_arr[i, 4])
                fgs.append(batch_results_arr[i, 6])
                bgs.append(batch_results_arr[i, 7])
                fpnlls.append(batch_results_arr[i, 8])
                bpnlls.append(batch_results_arr[i, 9])
                fnlls.append(batch_results_arr[i, 10])
                bnlls.append(batch_results_arr[i, 11])

            total_batches_seen += 1
            stage_batches_seen += 1

            if num_procs_found_solution > 0:
                to.set_grad_enabled(True)
                model.train()
                for grad_step in range(1, grad_steps + 1):
                    optimizer.zero_grad(set_to_none=False)
                    if f_traj:
                        (
                            f_loss,
                            f_avg_action_nll,
                            f_acc,
                        ) = loss_fn(f_traj, model)

                        local_opt_results[0] = f_loss.item()
                        local_opt_results[1] = f_acc

                        if b_traj:
                            (
                                b_loss,
                                b_avg_action_nll,
                                b_acc,
                            ) = loss_fn(b_traj, model)

                            local_opt_results[2] = b_loss.item()
                            local_opt_results[3] = b_acc

                            loss = f_loss + b_loss
                        else:
                            loss = f_loss

                        loss.backward()
                    else:
                        local_opt_results[:] = 0

                    # sync grads
                    all_grads_list = [
                        param.grad.view(-1) for param in model.parameters()
                    ]
                    all_grads = to.cat(all_grads_list)
                    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
                    all_grads.div_(num_procs_found_solution)
                    offset = 0
                    for param in model.parameters():
                        numel = param.numel()
                        param.grad.data.copy_(
                            all_grads[offset : offset + numel].view_as(param.grad.data)
                        )
                        offset += numel

                    optimizer.step()

                    # print batch losses/accs
                    if grad_step == grad_steps:
                        dist.all_reduce(local_opt_results, op=dist.ReduceOp.SUM)
                        if rank == 0:
                            batch_floss = (
                                local_opt_results[0].item() / num_procs_found_solution
                            )
                            batch_facc = (
                                local_opt_results[1].item() / num_procs_found_solution
                            )
                            batch_bloss = (
                                local_opt_results[2].item() / num_procs_found_solution
                            )
                            batch_bacc = (
                                local_opt_results[3].item() / num_procs_found_solution
                            )

                            flosses.append(batch_floss)
                            faccs.append(batch_facc)
                            print(f"floss: {batch_floss:.3f}")
                            print(f"facc: {batch_facc:.3f}")
                            if bidirectional:
                                blosses.append(batch_bloss)
                                baccs.append(batch_bacc)
                                print(f"bloss: {batch_bloss:.3f}")
                                print(f"bacc: {batch_bacc:.3f}")

            batch_avg = num_procs_found_solution / batch_size
            stage_avg += (batch_avg - stage_avg) / stage_batches_seen
            stage_problems_seen += batch_size

            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore", category=RuntimeWarning)

            if (
                train_loader.manual_advance
                and stage_problems_seen >= args.min_samples_per_stage
                and stage_avg >= args.min_stage_solve_ratio
            ):
                train_loader.goto_next_stage = True

            if rank == 0 and train_loader.goto_next_stage:
                stage_search_df = pd.DataFrame(
                    {
                        "id": pd.Series(ids, dtype=pd.UInt32Dtype()),
                        "time": pd.Series(times, dtype=pd.Float32Dtype()),
                        "len": pd.Series(lens, dtype=pd.UInt16Dtype()),
                        "fexp": pd.Series(fexps, dtype=pd.UInt16Dtype()),
                        "fg": pd.Series(fgs, dtype=pd.UInt16Dtype()),
                        "fpnll": pd.Series(fpnlls, dtype=pd.Float32Dtype()),
                        "fnll": pd.Series(fnlls, dtype=pd.Float32Dtype()),
                        "stage": pd.Series(
                            [old_stage for _ in range(stage_problems_seen)],
                            dtype=pd.UInt8Dtype(),
                        ),
                    }
                )
                stage_model_train_df = pd.DataFrame(
                    {
                        "floss": pd.Series(flosses, dtype=pd.Float32Dtype()),
                        "facc": pd.Series(faccs, dtype=pd.Float32Dtype()),
                        "stage": pd.Series(
                            [old_stage for _ in range(stage_problems_seen)],
                            dtype=pd.UInt8Dtype(),
                        ),
                    }
                )
                if bidirectional:
                    stage_search_df["bexp"] = pd.Series(bexps, dtype=pd.UInt16Dtype())
                    stage_search_df["bg"] = pd.Series(bgs, dtype=pd.UInt16Dtype())
                    stage_search_df["bpnll"] = pd.Series(
                        bpnlls, dtype=pd.Float32Dtype()
                    )
                    stage_search_df["bnll"] = pd.Series(bnlls, dtype=pd.Float32Dtype())

                    stage_model_train_df["bloss"] = pd.Series(
                        blosses, dtype=pd.Float32Dtype()
                    )
                    stage_model_train_df["bacc"] = pd.Series(
                        baccs, dtype=pd.Float32Dtype()
                    )

                print(
                    "----------------------------------------------------------------------------"
                )
                print(f"END STAGE {old_stage}\n")
                print_search_summary(
                    stage_search_df,
                    bidirectional,
                )
                print_model_train_summary(
                    stage_model_train_df,
                    bidirectional,
                )
                epoch_search_dfs.append(stage_search_df)
                epoch_model_train_dfs.append(stage_model_train_df)
                print(f"\nTime: {timer() - stage_start_time:.2f}")
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
                args,
                rank,
                agent,
                valid_loader,
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
                    best_models_log.write(
                        f"epoch {epoch} solved {valid_solved} exp {valid_total_expanded}\n"
                    )
                    agent.save_model("best_expanded", log=False)

                if valid_solved >= best_valid_solved:
                    best_valid_solved = valid_solved
                    print("Saving best model by solved")
                    best_models_log.write(
                        f"epoch {epoch} solved {valid_solved} exp {valid_total_expanded}\n"
                    )
                    agent.save_model("best_solved", log=False)
                best_models_log.flush()
                sys.stdout.flush()

            dist.barrier()

        epoch += 1
        # EPOCHS END
    if rank == 0:
        print("END TRAINING")
