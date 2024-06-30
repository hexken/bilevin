from argparse import Namespace
import pickle
import random
import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from tabulate import tabulate
import torch as to
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.utils import clip_grad_norm_ as clip_

from models.models import PolicyOrHeuristicModel
from search.agent import Agent
from search.loaders import AsyncProblemLoader, Problem
from search.utils import SearchResult, print_model_train_summary, print_search_summary
from test import test


def train(
    args: Namespace,
    rank: int,
    agent: Agent,
    train_loader: AsyncProblemLoader,
    valid_loader: AsyncProblemLoader,
    results_queue: mp.Queue,
):
    expansion_budget: int = args.train_expansion_budget
    max_expansion_budget: int = args.max_expansion_budget
    checkpoint_every_n_batches: int = args.checkpoint_every_n_batches

    best_models_log = (args.logdir / "best_models.txt").open("a")
    train_times_log = (args.logdir / "train_times.txt").open("a")

    model: PolicyOrHeuristicModel = agent.model
    bidirectional: bool = agent.is_bidirectional
    policy_based: bool = agent.has_policy
    heuristic_based: bool = agent.has_heuristic
    optimizer = agent.optimizer
    max_grad_norm = args.max_grad_norm
    loss_fn = agent.loss_fn

    best_valid_expanded = len(valid_loader) * expansion_budget + 1

    if args.checkpoint_path is None:


        last_epoch = 1
        epoch = 1
        batch = 1

        epoch_start_time = 0
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

            expansion_budget = chkpt_dict["current_exp_budget"]
            best_valid_expanded = chkpt_dict["best_valid_expanded"]
            epoch_start_time = timer() - chkpt_dict["time_in_stage"]
            epoch = chkpt_dict["epoch"]
            batch = chkpt_dict["batch"]
            # train_loader.load_state(chkpt_dict["loader_states"][rank])

        if rank == 0:
            print(
                "----------------------------------------------------------------------------"
            )
            print(
                f"Continuing from epoch {epoch} using checkpoint {args.checkpoint_path}"
            )
            print(
                "----------------------------------------------------------------------------"
            )

    old_checkpoint_path = args.logdir / f"dummy_chkpt"

    while epoch <= args.n_epochs:
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

        if rank == 0:
            print(
                "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
            )
            print(f"START EPOCH {epoch}")
            batch_start_time = timer()

        # expansion_budget = args.train_expansion_budget

        while True:
            problem = train_loader.get_problem()
            if problem is not None:
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

                if traj is not None:
                    f_traj,b_traj = traj
                    sol_len = len(f_traj)
                else:
                    f_traj = b_traj = None
                    sol_len = np.nan

                res = SearchResult(id=problem.id, time=end_time - start_time, fexp=n_forw_expanded,
                                   bexp=n_backw_expanded, len=sol_len, f_traj=f_traj, b_traj=b_traj)
                results_queue.put(res)

            else: # end batch
                dist.monitored_barrier()
                if rank == 0:
                    batch_buffer = []
                    while not results_queue.empty():
                        batch_buffer.append(results_queue.get())

                    batch_print_df = SearchResult.df(batch_buffer, policy_based, heuristic_based,
                                                     bidirectional)

                    print(f"\nBatch {batch}")
                    print(tabulate(
                            batch_print_df,
                            headers="keys",
                            tablefmt="psql",
                            showindex=False,
                            floatfmt=".2f",
                        ))
                    # update model :)
                    trajs = [(res.f_traj, res.b_traj) for res in batch_buffer if res.f_traj is
                             not None]
                    random.shuffle(trajs)

                    if len(trajs) > 0:
                        model.train()
                        for traj in trajs:
                            for grad_step in range(1, args.grad_steps + 1):
                                optimizer.zero_grad(set_to_none=False)
                                loss = loss_fn(traj[0], model)
                                if bidirectional:
                                    b_loss = loss_fn(traj[1], model)
                                    loss += b_loss
                                loss.backward()
                                optimizer.step()

                    batch_end_time = timer() - batch_start_time
                    batch_start_time = timer()
                    print(f"Batch time: {batch_end_time:.2f}s")
                dist.monitored_barrier()
                batch += 1

                if batch * args.batch_size >= len(train_loader): # end epoch
                    # log all search results
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
                                    flosses, dtype=pd.Float32Dtype(), index=batch
                                ),
                            },
                        )

                        if policy_based:
                            stage_search_df["fap"] = pd.Series(faps, dtype=pd.Float32Dtype())
                            stage_search_df["facc"] = pd.Series(faccs, dtype=pd.Float32Dtype())
                        if heuristic_based:
                            stage_search_df["fhe"] = pd.Series(fhes, dtype=pd.Float32Dtype())

                        if bidirectional:
                            stage_search_df["bexp"] = pd.Series(bexps, dtype=pd.UInt32Dtype())
                            stage_search_df["bg"] = pd.Series(bgs, dtype=pd.UInt32Dtype())

                            stage_model_train_df["bloss"] = pd.Series(
                                blosses, dtype=pd.Float32Dtype(), index=batch
                            )
                            if policy_based:
                                stage_search_df["bacc"] = pd.Series(
                                    baccs, dtype=pd.Float32Dtype()
                                )
                                stage_search_df["bap"] = pd.Series(
                                    baps, dtype=pd.Float32Dtype()
                                )
                            if heuristic_based:
                                stage_search_df["bhe"] = pd.Series(
                                    bhes, dtype=pd.Float32Dtype()
                                )

                        print(f"\nEND EPOCH {epoch}\n")
                        agent.save_model("latest", log=False)
                        with (args.logdir / f"search_train_e{epoch}.pkl").open(
                            "wb"
                        ) as f:
                            pickle.dump(stage_search_df, f)
                        with (args.logdir / f"model_train_e{epoch}.pkl").open(
                            "wb"
                        ) as f:
                            pickle.dump(stage_model_train_df, f)

                        total_epoch_time = timer() - epoch_start_time

                        train_times_log.write(f"{epoch} {total_epoch_time:.2f}\n")
                        train_times_log.flush()

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
                        print(f"\nTime: {total_epoch_time:.2f}s")
                        print(
                            "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                        )

                        # Validation checks
                        if (
                            args.validate_every_n_batch > 0
                            and batch % args.validate_every_n_batch == 0) or done_training:
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
                                    batch=batch,
                                )

                                if rank == 0:
                                    (
                                        valid_solved,
                                        valid_total_expanded,
                                        valid_total_time,
                                    ) = valid_results

                                    epoch_start_time += valid_total_time

                                    if valid_total_expanded <= best_valid_expanded:
                                        best_valid_expanded = valid_total_expanded
                                        print("Saving best model")
                                        best_models_log.write(
                                            f"batch {batch} solved {valid_solved} exp {valid_total_expanded}\n"
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
                        if (batch % checkpoint_every_n_batches == 0) or done_training or validate:
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
                                    "flosses": flosses,
                                    "faccs": faccs,
                                    "blosses": blosses,
                                    "baccs": baccs,
                                    "current_exp_budget": expansion_budget,
                                    "best_valid_expanded": best_valid_expanded,
                                    "time_in_stage": timer() - epoch_start_time,
                                    "epoch": epoch,
                                    "batch": batch,
                                    "loader_states": loader_states,
                                    "done_training": done_training,
                                    "reduce_lr": reduce_lr,
                                }
                                new_checkpoint_path = args.logdir / f"checkpoint_b{batch}.pkl"
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
