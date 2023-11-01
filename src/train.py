from argparse import Namespace
from pathlib import Path
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
    args: Namespace,
    rank: int,
    agent: Agent,
    train_loader: ProblemLoader,
    valid_loader: ProblemLoader,
):
    logdir = args.logdir
    batch_size = args.world_size
    time_budget = args.time_budget
    expansion_budget = args.train_expansion_budget
    grad_steps = args.grad_steps

    best_models_log = logdir / "best_models.txt"
    if rank == 0:
        best_models_log = best_models_log.open("a")

    bidirectional = agent.bidirectional
    model = agent.model
    optimizer = agent.optimizer
    loss_fn = agent.loss_fn

    solved_flag = to.zeros(1, dtype=to.uint8)
    local_opt_results = to.zeros(4, dtype=to.float64)

    world_search_results = [
        to.zeros(len(search_result_header), dtype=to.float64) for _ in range(batch_size)
    ]

    batches_seen = 0

    num_valid_problems = len(valid_loader)
    max_valid_expanded = num_valid_problems * expansion_budget
    best_valid_expanded = max_valid_expanded + 1

    if args.checkpoint_path is None:
        for param in model.parameters():
            if not param.grad:
                param.grad = to.zeros_like(param)
        ids = []
        times = []
        lens = []
        fexps = []
        bexps = []
        fgs = []
        bgs = []
        fpnlls = []
        bpnlls = []

        flosses = []
        faccs = []
        blosses = []
        baccs = []

        loader_state = None
        stage_problems_seen = 0
        stage_problems_this_budget = 0
    else:
        chkpt_dict = pickle.load(args.checkpoint_path.open("rb"))
        model.load_state_dict(chkpt_dict["model_state"])
        optimizer.load_state_dict(chkpt_dict["optimizer_state"])
        ids = chkpt_dict["ids"]
        times = chkpt_dict["times"]
        lens = chkpt_dict["lens"]
        fexps = chkpt_dict["fexps"]
        bexps = chkpt_dict["bexps"]
        fgs = chkpt_dict["fgs"]
        bgs = chkpt_dict["bgs"]
        fpnlls = chkpt_dict["fpnlls"]
        bpnlls = chkpt_dict["bpnlls"]

        flosses = chkpt_dict["flosses"]
        faccs = chkpt_dict["faccs"]
        blosses = chkpt_dict["blosses"]
        baccs = chkpt_dict["baccs"]

        batches_seen = chkpt_dict["batches_seen"]
        stage_problems_seen = chkpt_dict["stage_problems_seen"]
        stage_problems_this_budget = chkpt_dict["stage_problems_this_budget"]

        loader_state = chkpt_dict["loader_state"]
        stage_problems_seen = chkpt_dict["stage_problems_seen"]
        stage_problems_this_budget = chkpt_dict["stage_problems_this_budget"]
        expansion_budget = chkpt_dict["current_exp_budget"]
        train_loader.load_state(loader_state, rank)

        if rank == 0:
            print(
                "----------------------------------------------------------------------------"
            )
            print(
                f"Continuing from stage {train_loader.stage} using checkpoint {args.checkpoint_path}"
            )
            print(
                "----------------------------------------------------------------------------"
            )

    stage_start_time = 0

    old_checkpoint_path = logdir / f"dummy_chkpt"

    old_stage = train_loader.stage

    for problem in train_loader:
        if old_stage != train_loader.stage:
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

            flosses = []
            faccs = []
            blosses = []
            baccs = []

            stage_problems_seen = 0
            stage_problems_this_budget = 0

            if rank == 0:
                print(
                    "----------------------------------------------------------------------------"
                )
                print(f"START STAGE {old_stage}")
                expansion_budget = args.train_expansion_budget
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
        sol_len = np.nan if not traj else len(traj[0])

        # to clear the domain cache
        if bidirectional:
            problem.domain.reset()

        local_search_results = to.zeros(len(search_result_header), dtype=to.float64)
        local_search_results[:] = np.nan
        local_search_results[0] = problem.id
        local_search_results[1] = end_time - start_time
        local_search_results[2] = n_forw_expanded + n_backw_expanded
        local_search_results[3] = n_forw_expanded
        local_search_results[4] = n_backw_expanded
        local_search_results[5] = sol_len

        if traj:
            solved_flag[0] = 1
            f_traj, b_traj = traj
            local_search_results[6] = f_traj.partial_g_cost
            local_search_results[8] = -1 * f_traj.partial_log_prob
            if b_traj:
                local_search_results[7] = b_traj.partial_g_cost
                local_search_results[9] = -1 * b_traj.partial_log_prob

        else:
            local_search_results[6:] = np.nan
            f_traj = b_traj = None
            solved_flag[0] = 0

        dist.all_gather(world_search_results, local_search_results)
        dist.all_reduce(solved_flag, op=dist.ReduceOp.SUM)
        num_procs_found_solution = solved_flag[0].item()

        batches_seen += 1

        world_batch_results_t = to.stack(world_search_results, dim=0)
        batch_results_arr = world_batch_results_t.numpy()
        batch_print_df = pd.DataFrame(batch_results_arr, columns=search_result_header)
        for col in int_columns:
            batch_print_df[col] = batch_print_df[col].astype(pd.UInt32Dtype())
        batch_print_df = batch_print_df.sort_values("exp")

        if rank == 0:
            print(f"\nBatch {batches_seen}")

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

        if num_procs_found_solution > 0:
            to.set_grad_enabled(True)
            model.train()
            for grad_step in range(1, grad_steps + 1):
                optimizer.zero_grad(set_to_none=False)
                if f_traj:
                    (
                        f_loss,
                        _,
                        f_acc,
                    ) = loss_fn(f_traj, model)

                    local_opt_results[0] = f_loss.item()
                    local_opt_results[1] = f_acc

                    if b_traj:
                        (
                            b_loss,
                            _,
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
                all_grads_list = [param.grad.view(-1) for param in model.parameters()]
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

        stage_problems_seen += batch_size
        stage_problems_this_budget += batch_size

        if len(lens) >= args.n_solve_ratio:
            stage_solve_ratio = (
                ~np.isnan(np.array(lens[-args.n_solve_ratio :]))
            ).mean()
        else:
            stage_solve_ratio = 0

        if (
            train_loader.manual_advance
            and stage_problems_seen >= args.min_samples_per_stage
            and stage_solve_ratio >= args.min_solve_ratio
        ):
            train_loader.goto_next_stage = True
        elif (
            args.increase_budget
            and stage_problems_this_budget >= args.n_solve_ratio
            and stage_solve_ratio <= args.min_solve_ratio
        ):
            old_budget = expansion_budget
            expansion_budget *= 2
            stage_problems_this_budget = 0
            if rank == 0:
                print(
                    "----------------------------------------------------------------------------"
                )
                print(f"Budget increased from {old_budget} to {expansion_budget}")
                print(
                    "----------------------------------------------------------------------------"
                )

        if rank == 0 and train_loader.goto_next_stage:
            stage_search_df = pd.DataFrame(
                {
                    "id": pd.Series(ids, dtype=pd.UInt32Dtype()),
                    "time": pd.Series(times, dtype=pd.Float32Dtype()),
                    "len": pd.Series(lens, dtype=pd.UInt16Dtype()),
                    "fexp": pd.Series(fexps, dtype=pd.UInt16Dtype()),
                    "fg": pd.Series(fgs, dtype=pd.UInt16Dtype()),
                    "fpnll": pd.Series(fpnlls, dtype=pd.Float32Dtype()),
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
                stage_search_df["bpnll"] = pd.Series(bpnlls, dtype=pd.Float32Dtype())
                stage_model_train_df["bloss"] = pd.Series(
                    blosses, dtype=pd.Float32Dtype()
                )
                stage_model_train_df["bacc"] = pd.Series(baccs, dtype=pd.Float32Dtype())

            print(
                "----------------------------------------------------------------------------"
            )
            print(f"END STAGE {old_stage}\n")
            agent.save_model("latest", log=False)
            with (logdir / f"search_train_s{old_stage}.pkl").open("wb") as f:
                pickle.dump(stage_search_df, f)
            with (logdir / f"model_train_s{old_stage}.pkl").open("wb") as f:
                pickle.dump(stage_model_train_df, f)
            print_search_summary(
                stage_search_df,
                bidirectional,
            )
            print_model_train_summary(
                stage_model_train_df,
                bidirectional,
            )
            print(f"\nTime: {timer() - stage_start_time:.2f}")
            print(
                "----------------------------------------------------------------------------"
            )
        sys.stdout.flush()

        if (
            batches_seen >= args.batch_begin_validate
            and batches_seen % args.validate_every == 0
        ):
            if rank == 0:
                print(
                    "----------------------------------------------------------------------------"
                )
                print("VALIDATION")

            dist.monitored_barrier()

            valid_results = test(
                args,
                rank,
                agent,
                valid_loader,
                print_results=False,
                batch=batches_seen,
            )

            if rank == 0:
                (
                    valid_solved,
                    valid_total_expanded,
                ) = valid_results

                if valid_total_expanded <= best_valid_expanded:
                    best_valid_expanded = valid_total_expanded
                    print("Saving best model")
                    best_models_log.write(
                        f"batch {batches_seen} solved {valid_solved} exp {valid_total_expanded}\n"
                    )
                    agent.save_model("best_expanded", log=False)

                best_models_log.flush()
                print(
                    "----------------------------------------------------------------------------"
                )
                sys.stdout.flush()

            dist.monitored_barrier()

        # Checkpoint
        if batches_seen % args.checkpoint_every == 0:
            # add indices of all ranks to loader_state
            loader_state = train_loader.get_state()
            indices = [None for _ in range(args.world_size)]
            dist.gather_object(
                loader_state["indices"], indices if rank == 0 else None, dst=0
            )
            loader_state = train_loader.get_state()
            loader_state["indices"] = indices

            if rank == 0:
                chkpt_dict = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "ids": ids,
                    "times": times,
                    "lens": lens,
                    "fexps": fexps,
                    "bexps": bexps,
                    "fgs": fgs,
                    "bgs": bgs,
                    "fpnlls": fpnlls,
                    "bpnlls": bpnlls,
                    "flosses": flosses,
                    "faccs": faccs,
                    "blosses": blosses,
                    "baccs": baccs,
                    "current_exp_budget": expansion_budget,
                    "best_valid_expanded": best_valid_expanded,
                    "stage_problems_seen": stage_problems_seen,
                    "stage_problems_this_budget": stage_problems_this_budget,
                    "batches_seen": batches_seen,
                    "loader_state": loader_state,
                }
                new_checkpoint_path = logdir / f"checkpoint_b{batches_seen}.pkl"
                with new_checkpoint_path.open("wb") as f:
                    pickle.dump(chkpt_dict, f)
                if not args.keep_all_checkpoints:
                    old_checkpoint_path.unlink(missing_ok=True)
                    old_checkpoint_path = new_checkpoint_path
