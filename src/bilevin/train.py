from argparse import Namespace
import pickle
import sys
from timeit import default_timer as timer
import tracemalloc

import numpy as np
import pandas as pd
from tabulate import tabulate
from test import test
import torch as to
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_ as clip_

from loaders import ProblemLoader
from search.approx_ff import ApproxFF
from models.models import SuperModel
from search.agent import Agent
from search.utils import (
    int_columns,
    print_model_train_summary,
    print_search_summary,
    search_result_header,
)
from utils import display_top


def train(
    args: Namespace,
    rank: int,
    agent: Agent,
    train_loader: ProblemLoader,
    valid_loader: ProblemLoader,
):
    # tracemalloc.start()
    batch_size: int = args.world_size

    expansion_budget: int = args.train_expansion_budget
    max_expansion_budget: int = args.max_expansion_budget

    min_batches_per_stage: int = args.min_batches_per_stage
    max_batches_per_stage: int = args.max_batches_per_stage
    min_batches_final_stage: int = args.min_batches_final_stage
    max_batches_final_stage: int = args.max_batches_final_stage
    n_batch_tail: int = args.n_batch_tail

    min_solve_ratio_stage: float = args.min_solve_ratio_stage
    min_solve_ratio_final_stage: float = args.min_solve_ratio_final_stage
    min_solve_ratio_exp: float = args.min_solve_ratio_exp

    batch_begin_validate: int = args.batch_begin_validate
    validate_every_n_batch: int = args.validate_every_n_batch
    stage_begin_validate: int = args.stage_begin_validate
    validate_every_n_stage: bool = args.validate_every_n_stage
    validate_every_epoch: bool = args.validate_every_epoch

    checkpoint_every_n_batch: int = args.checkpoint_every_n_batch

    reduce_lr = False

    best_models_log = (args.logdir / "best_models.txt").open("a")
    train_times_log = (args.logdir / "train_times.txt").open("a")

    model: SuperModel = agent.model
    bidirectional: bool = agent.is_bidirectional
    policy_based: bool = agent.has_policy
    heuristic_based: bool = agent.has_heuristic
    optimizer = agent.model.optimizer
    max_grad_norm = args.max_grad_norm
    loss_fn = agent.model.loss_fn

    solved_flag = to.zeros(1, dtype=to.uint8)
    local_opt_results = to.zeros(4, dtype=to.float64)

    world_search_results = [
        to.zeros(len(search_result_header), dtype=to.float64) for _ in range(batch_size)
    ]
    local_search_results = to.zeros(len(search_result_header), dtype=to.float64)

    num_valid_problems = len(valid_loader)
    max_valid_expanded = num_valid_problems * expansion_budget
    best_valid_expanded = max_valid_expanded + 1

    for param in model.parameters():
        if not param.grad:
            param.grad = to.zeros_like(param)

    if args.checkpoint_path is None:
        ids = []
        times = []
        lens = []
        fexps = []
        bexps = []
        fgs = []
        bgs = []
        faps = []  # forward avg. action probs
        baps = []
        fhes = []  # forward avg. abs. heuristic errors
        bhes = []

        flosses = []
        faccs = []
        blosses = []
        baccs = []
        batches = []

        final_stage_epoch = 1 if train_loader.n_stages == 1 else 0
        batches_seen = 0
        batch_solve_ratios = []

        stage_start_time = 0
        stage_batches_seen = 0
        stage_batches_this_budget = 0
    else:
        with args.checkpoint_path.open("rb") as f:
            chkpt_dict = to.load(f)
            optimizer.load_state_dict(chkpt_dict["optimizer_state"])
            ids = chkpt_dict["ids"]
            times = chkpt_dict["times"]
            lens = chkpt_dict["lens"]
            fexps = chkpt_dict["fexps"]
            bexps = chkpt_dict["bexps"]
            fgs = chkpt_dict["fgs"]
            bgs = chkpt_dict["bgs"]
            faps = chkpt_dict["faps"]
            baps = chkpt_dict["baps"]
            fhes = chkpt_dict["fhes"]
            bhes = chkpt_dict["bhes"]

            flosses = chkpt_dict["flosses"]
            faccs = chkpt_dict["faccs"]
            blosses = chkpt_dict["blosses"]
            baccs = chkpt_dict["baccs"]
            batches = chkpt_dict["batches"]
            min_batches_per_stage = chkpt_dict["min_batches_per_stage"]
            max_batches_per_stage = chkpt_dict["max_batches_per_stage"]
            min_batches_final_stage = chkpt_dict["min_batches_final_stage"]
            max_batches_final_stage = chkpt_dict["max_batches_final_stage"]

            expansion_budget = chkpt_dict["current_exp_budget"]
            best_valid_expanded = chkpt_dict["best_valid_expanded"]
            stage_start_time = timer() - chkpt_dict["time_in_stage"]
            stage_batches_seen = chkpt_dict["stage_batches_seen"]
            stage_batches_this_budget = chkpt_dict["stage_batches_this_budget"]
            final_stage_epoch = chkpt_dict["final_stage_epoch"]
            batches_seen = chkpt_dict["batches_seen"]
            n_batch_tail = chkpt_dict["n_batch_tail"]
            batch_solve_ratios = chkpt_dict["batch_solve_ratios"]
            reduce_lr = chkpt_dict["reduce_lr"]
            train_loader.load_state(chkpt_dict["loader_states"][rank])

        if rank == 0:
            estr = "" if final_stage_epoch == 0 else f" epoch {final_stage_epoch}"
            print(
                "----------------------------------------------------------------------------"
            )
            print(
                f"Continuing from stage {train_loader.stage}{estr} using checkpoint {args.checkpoint_path}"
            )
            print(
                "----------------------------------------------------------------------------"
            )

    old_checkpoint_path = args.logdir / f"dummy_chkpt"

    old_stage = train_loader.stage
    done_training = False

    for problem in train_loader:
        batch_start_time = timer()
        if old_stage != train_loader.stage or train_loader.repeat_stage:
            train_loader.repeat_stage = False
            stage_start_time = timer()
            old_stage = train_loader.stage

            if args.min_batches_per_stage == -1:
                min_batches_per_stage = len(train_loader.stage_problems)
            if args.max_batches_per_stage == -1:
                max_batches_per_stage = len(train_loader.stage_problems)

            if args.min_batches_final_stage == -1:
                min_batches_final_stage = len(train_loader.stage_problems)
            if args.max_batches_final_stage == -1:
                max_batches_final_stage = len(train_loader.stage_problems)

            if args.n_batch_tail == -1:
                n_batch_tail = len(train_loader.stage_problems)

            stage_batches_seen = 0
            batch_solve_ratios = []
            stage_batches_this_budget = 0

            ids = []
            times = []
            lens = []
            fexps = []
            bexps = []
            fgs = []
            bgs = []
            faps = []
            baps = []
            fhes = []
            bhes = []

            flosses = []
            faccs = []
            blosses = []
            baccs = []
            batches = []

            if rank == 0:
                estr = "" if final_stage_epoch == 0 else f" EPOCH {final_stage_epoch}"
                print(
                    "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
                )
                print(f"START STAGE {old_stage}{estr}")
                expansion_budget = args.train_expansion_budget
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
            time_budget=args.time_budget,
        )
        end_time = timer()
        sol_len = np.nan if not traj else len(traj[0])

        local_search_results[:] = np.nan
        local_search_results[0] = problem.id
        local_search_results[1] = end_time - start_time
        local_search_results[2] = n_forw_expanded
        local_search_results[3] = n_backw_expanded
        local_search_results[4] = sol_len
        del problem

        if traj:
            solved_flag[0] = 1
            f_traj, b_traj = traj
            local_search_results[5] = f_traj.partial_g_cost
            local_search_results[6] = f_traj.avg_action_prob
            local_search_results[7] = f_traj.avg_h_abs_error
            if b_traj:
                local_search_results[8] = b_traj.partial_g_cost
                local_search_results[9] = b_traj.avg_action_prob
                local_search_results[10] = b_traj.avg_h_abs_error

        else:
            f_traj = b_traj = None
            solved_flag[0] = 0

        dist.all_gather(world_search_results, local_search_results)
        dist.all_reduce(solved_flag, op=dist.ReduceOp.SUM)
        num_procs_found_solution = solved_flag[0].item()
        batch_solve_ratios.append(num_procs_found_solution / batch_size)

        batches_seen += 1

        world_batch_results_t = to.stack(world_search_results, dim=0)
        batch_results_arr = world_batch_results_t.numpy()
        del world_batch_results_t
        batch_print_df = pd.DataFrame(batch_results_arr, columns=search_result_header)
        for col in int_columns:
            batch_print_df[col] = batch_print_df[col].astype(pd.UInt32Dtype())
        if bidirectional:
            exp = batch_print_df["fexp"] + batch_print_df["bexp"]
        else:
            exp = batch_print_df["fexp"]
        batch_print_df.insert(2, "exp", exp)
        batch_print_df = batch_print_df.sort_values("exp")

        if rank == 0:
            print(f"\nBatch {batches_seen}")

        if rank == 0:
            if not policy_based:
                batch_print_df = batch_print_df.drop(
                    columns=["facc", "fap", "bap", "bacc"], errors="ignore"
                )
            if not heuristic_based:
                batch_print_df = batch_print_df.drop(
                    columns=["fhe", "bhe"], errors="ignore"
                )
            if not bidirectional:
                batch_print_df = batch_print_df.drop(
                    columns=["bexp", "bg", "bacc", "bap", "bhe"], errors="ignore"
                )
            print(
                tabulate(
                    batch_print_df,
                    headers="keys",
                    tablefmt="psql",
                    showindex=False,
                    floatfmt=".2f",
                )
            )
        del batch_print_df

        for i in range(batch_size):
            ids.append(batch_results_arr[i, 0])
            times.append(batch_results_arr[i, 1])
            fexps.append(batch_results_arr[i, 2])
            bexps.append(batch_results_arr[i, 3])
            lens.append(batch_results_arr[i, 4])
            fgs.append(batch_results_arr[i, 5])
            faps.append(batch_results_arr[i, 6])
            fhes.append(batch_results_arr[i, 7])
            bgs.append(batch_results_arr[i, 8])
            baps.append(batch_results_arr[i, 9])
            bhes.append(batch_results_arr[i, 10])
        del batch_results_arr

        if num_procs_found_solution > 0:
            to.set_grad_enabled(True)
            model.train()
            for grad_step in range(1, args.grad_steps + 1):
                optimizer.zero_grad(set_to_none=False)
                if isinstance(agent, ApproxFF):
                    if traj:
                        loss = loss_fn(f_traj, b_traj, model)
                        local_opt_results[0] = loss.item()
                    else:
                        local_opt_results[:] = 0
                else:
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
                del all_grads_list
                dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
                all_grads.div_(num_procs_found_solution)
                if max_grad_norm > 0:
                    clip_(all_grads, max_grad_norm)
                offset = 0
                for param in model.parameters():
                    numel = param.numel()
                    param.grad.data.copy_(
                        all_grads[offset : offset + numel].view_as(param.grad.data)
                    )
                    offset += numel
                del all_grads

                optimizer.step()

                # print batch losses/accs
                if grad_step == args.grad_steps:
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

                        batches.append(batches_seen)
                        flosses.append(batch_floss)
                        print(f"floss: {batch_floss:.3f}")
                        if policy_based:
                            faccs.append(batch_facc)
                            print(f"facc: {batch_facc:.3f}")
                        if bidirectional:
                            blosses.append(batch_bloss)
                            print(f"bloss: {batch_bloss:.3f}")
                            if policy_based:
                                baccs.append(batch_bacc)
                                print(f"bacc: {batch_bacc:.3f}")

        del traj, f_traj, b_traj
        stage_batches_seen += 1
        stage_batches_this_budget += 1

        # if batches_seen % 5 == 0:
        #     if rank == 0:
        #         display_top(tracemalloc.take_snapshot(), limit=25)

        # Stage completion checks
        solve_ratio = None
        if train_loader.stage < train_loader.n_stages:
            if stage_batches_seen >= min_batches_per_stage:
                if min_solve_ratio_stage > 0:
                    if len(batch_solve_ratios) >= n_batch_tail:
                        solve_ratio = (
                            sum(batch_solve_ratios[-n_batch_tail:]) / n_batch_tail
                        )
                        if solve_ratio >= min_solve_ratio_stage:
                            train_loader.stage_complete = True
                else:
                    train_loader.stage_complete = True
            if (
                stage_batches_seen >= max_batches_per_stage
                and max_batches_per_stage > 0
            ):
                train_loader.stage_complete = True
        else:  # final stage
            if stage_batches_seen >= min_batches_final_stage:
                if min_solve_ratio_final_stage > 0:
                    if len(batch_solve_ratios) >= n_batch_tail:
                        solve_ratio = (
                            sum(batch_solve_ratios[-n_batch_tail:]) / n_batch_tail
                        )
                        if solve_ratio >= min_solve_ratio_final_stage:
                            train_loader.stage_complete = True
                else:
                    train_loader.stage_complete = True
            if (
                stage_batches_seen >= max_batches_final_stage
                and max_batches_final_stage > 0
            ):
                train_loader.stage_complete = True

        # Increase expansion budget checks
        if (
            min_solve_ratio_exp > 0
            and expansion_budget < max_expansion_budget
            and not train_loader.stage_complete
            and stage_batches_this_budget >= n_batch_tail
        ):
            if solve_ratio is None:
                solve_ratio = sum(batch_solve_ratios[-n_batch_tail:]) / n_batch_tail
            if solve_ratio < min_solve_ratio_exp:
                old_budget = expansion_budget
                expansion_budget *= 2
                stage_batches_this_budget = 0
                if rank == 0:
                    print(
                        "----------------------------------------------------------------------------"
                    )
                    print(f"Budget increased from {old_budget} to {expansion_budget}")
                    print(
                        "----------------------------------------------------------------------------"
                    )
        batch_end_time = timer() - batch_start_time
        if rank == 0:
            print(f"Batch time: {batch_end_time:.2f}s")

        if train_loader.stage_complete:
            if rank == 0:
                stage_search_df = pd.DataFrame(
                    {
                        "id": pd.Series(ids, dtype=pd.UInt32Dtype()),
                        "time": pd.Series(times, dtype=pd.Float32Dtype()),
                        "len": pd.Series(lens, dtype=pd.UInt32Dtype()),
                        "fexp": pd.Series(fexps, dtype=pd.UInt32Dtype()),
                        "fg": pd.Series(fgs, dtype=pd.UInt32Dtype()),
                    }
                )
                stage_model_train_df = pd.DataFrame(
                    {
                        "floss": pd.Series(
                            flosses, dtype=pd.Float32Dtype(), index=batches
                        ),
                    },
                )

                if policy_based:
                    stage_search_df["fap"] = pd.Series(faps, dtype=pd.Float32Dtype())
                    stage_model_train_df["facc"] = pd.Series(
                        faccs, dtype=pd.Float32Dtype(), index=batches
                    )
                if heuristic_based:
                    stage_search_df["fhe"] = pd.Series(fhes, dtype=pd.Float32Dtype())

                if bidirectional:
                    stage_search_df["bexp"] = pd.Series(bexps, dtype=pd.UInt32Dtype())
                    stage_search_df["bg"] = pd.Series(bgs, dtype=pd.UInt32Dtype())

                    stage_model_train_df["bloss"] = pd.Series(
                        blosses, dtype=pd.Float32Dtype(), index=batches
                    )
                    if policy_based:
                        stage_model_train_df["bacc"] = pd.Series(
                            baccs, dtype=pd.Float32Dtype(), index=batches
                        )
                        stage_search_df["bap"] = pd.Series(
                            baps, dtype=pd.Float32Dtype()
                        )
                    if heuristic_based:
                        stage_search_df["bhe"] = pd.Series(
                            bhes, dtype=pd.Float32Dtype()
                        )

                estr = "" if final_stage_epoch == 0 else f" EPOCH {final_stage_epoch}"
                print(f"\nEND STAGE {old_stage}{estr}\n")
                estr = "" if final_stage_epoch == 0 else f"_e{final_stage_epoch}"
                agent.save_model("latest", log=False)
                with (args.logdir / f"search_train_s{old_stage}{estr}.pkl").open(
                    "wb"
                ) as f:
                    pickle.dump(stage_search_df, f)
                with (args.logdir / f"model_train_s{old_stage}{estr}.pkl").open(
                    "wb"
                ) as f:
                    pickle.dump(stage_model_train_df, f)

                total_stage_time = timer() - stage_start_time

                e = 1 if final_stage_epoch == 0 else final_stage_epoch
                train_times_log.write(f"{old_stage} {e} {total_stage_time:.2f}\n")

                print_search_summary(
                    stage_search_df,
                    bidirectional,
                )
                print_model_train_summary(
                    stage_model_train_df,
                    bidirectional,
                    policy_based,
                )
                del stage_search_df, stage_model_train_df
                print(f"\nTime: {total_stage_time:.2f}s")
                print(
                    "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                )

            if train_loader.stage + 1 == train_loader.n_stages:
                # about to begin 1st epoch of final stage when n_stages > 1
                if args.n_final_stage_epochs > 0:
                    final_stage_epoch = 1
                else:
                    done_training = True
            elif train_loader.stage == train_loader.n_stages:
                # about to repeat final stage
                if final_stage_epoch < args.n_final_stage_epochs:
                    train_loader.repeat_stage = True
                    final_stage_epoch += 1
                else:
                    done_training = True

        sys.stdout.flush()

        # Validation checks
        if (
            validate_every_n_batch > 0
            and batches_seen % validate_every_n_batch == 0
            and batches_seen >= batch_begin_validate
        ):
            validate = True
        elif (
            validate_every_n_stage > 0
            and train_loader.stage_complete
            and train_loader.stage % validate_every_n_stage == 0
            and train_loader.stage >= stage_begin_validate
        ):
            validate = True
        elif train_loader.repeat_stage and validate_every_epoch:
            validate = True
        elif done_training:
            validate = True
        else:
            validate = False

        if validate:
            if rank == 0:
                print(
                    ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                )
                print("VALIDATION")
                sys.stdout.flush()

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
                    valid_total_time,
                ) = valid_results

                stage_start_time += valid_total_time

                if valid_total_expanded <= best_valid_expanded:
                    best_valid_expanded = valid_total_expanded
                    print("Saving best model")
                    best_models_log.write(
                        f"batch {batches_seen} solved {valid_solved} exp {valid_total_expanded}\n"
                    )
                    agent.save_model("best_expanded", log=False)

                if (
                    reduce_lr is not None
                    and args.solve_ratio_reduce_lr > 0
                    and valid_solved / num_valid_problems >= args.solve_ratio_reduce_lr
                ):
                    reduce_lr = True
                    print("Reducing learning rate")

                print(
                    ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                )
                sys.stdout.flush()

            dist.monitored_barrier()
            if reduce_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= args.lr_decay
                reduce_lr = None

        # Checkpoint
        if (batches_seen % checkpoint_every_n_batch == 0) or done_training or validate:
            ts = timer()
            loader_states = [None] * args.world_size
            dist.gather_object(
                train_loader.get_state(), loader_states if rank == 0 else None, dst=0
            )

            if rank == 0:
                chkpt_dict = {
                    "stage": old_stage,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "ids": ids,
                    "times": times,
                    "lens": lens,
                    "fexps": fexps,
                    "bexps": bexps,
                    "fgs": fgs,
                    "bgs": bgs,
                    "faps": faps,
                    "baps": baps,
                    "fhes": fhes,
                    "bhes": bhes,
                    "batches": batches,
                    "flosses": flosses,
                    "faccs": faccs,
                    "blosses": blosses,
                    "baccs": baccs,
                    "current_exp_budget": expansion_budget,
                    "best_valid_expanded": best_valid_expanded,
                    "time_in_stage": timer() - stage_start_time,
                    "stage_batches_seen": stage_batches_seen,
                    "n_batch_tail": n_batch_tail,
                    "batch_solve_ratios": batch_solve_ratios,
                    "stage_batches_this_budget": stage_batches_this_budget,
                    "min_batches_per_stage": min_batches_per_stage,
                    "max_batches_per_stage": max_batches_per_stage,
                    "min_batches_final_stage": min_batches_final_stage,
                    "max_batches_final_stage": max_batches_final_stage,
                    "final_stage_epoch": final_stage_epoch,
                    "batches_seen": batches_seen,
                    "loader_states": loader_states,
                    "done_training": done_training,
                    "reduce_lr": reduce_lr,
                }
                new_checkpoint_path = args.logdir / f"checkpoint_b{batches_seen}.pkl"
                with new_checkpoint_path.open("wb") as f:
                    to.save(chkpt_dict, f)
                del loader_states, chkpt_dict
                if not args.keep_all_checkpoints:
                    old_checkpoint_path.unlink(missing_ok=True)
                    old_checkpoint_path = new_checkpoint_path
                print(
                    f"\nCheckpoint saved to {new_checkpoint_path.name}, took {timer() - ts:.2f}s"
                )

        if done_training:
            dist.monitored_barrier()
            break
