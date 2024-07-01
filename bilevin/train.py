from argparse import Namespace
from copy import copy
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

from search.agent import Agent
from search.loaders import AsyncProblemLoader
from search.utils import (
    Result,
    ResultsLog,
    print_model_train_summary,
    print_search_summary,
)
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
    checkpoint_every_n_batches: int = args.checkpoint_every_n_batches

    best_models_log = (args.logdir / "best_models.txt").open("a")
    simple_log = (args.logdir / "simple_log.txt").open("a")

    best_valid_expanded = len(valid_loader) * expansion_budget + 1

    if args.checkpoint_path is None:
        results = ResultsLog(
            None, agent.has_policy, agent.has_heuristic, agent.is_bidirectional
        )
        epoch = 0
        batch = 0
        done_epoch = False
        train_start_time = timer()
    else:
        with args.checkpoint_path.open("rb") as f:
            checkpoint_data = to.load(f)
            agent.optimizer.load_state_dict(checkpoint_data["optimizer_state"])
            results = ResultsLog(
                checkpoint_data["search_results"],
                agent.has_policy,
                agent.has_heuristic,
                agent.is_bidirectional,
            )

            best_valid_expanded = checkpoint_data["best_valid_expanded"]
            epoch_start_time = (
                timer() - checkpoint_data["time_in_epoch"]
            )  # todo check these
            train_start_time = timer() - checkpoint_data["time_in_training"]
            epoch = checkpoint_data["epoch"]
            batch = checkpoint_data[
                "batch"
            ]  # checkpoint records how many batches completed
            done_epoch = checkpoint_data["done_epoch"]
            # train_loader.load_state(chkpt_dict["loader_states"][rank])

        if rank == 0:
            print(
                "----------------------------------------------------------------------------"
            )
            print(f"Continuing epoch {epoch} using checkpoint {args.checkpoint_path}")
            print(
                "----------------------------------------------------------------------------"
            )

    old_checkpoint_path = args.logdir / f"dummy_chkpt"

    while epoch <= args.n_epochs:

        if rank == 0:
            print(
                ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            )
            print(f"START EPOCH {epoch}")

        done_batch = True
        while True:
            if done_batch:
                batch_start_time = timer()
                done_batch = False
                batch += 1
                if done_epoch or epoch == 0:
                    epoch_start_time = timer()
                    done_epoch = False
                    epoch += 1
                    if rank == 0:
                        train_loader.init_indexer(shuffle=args.shuffle)
                if rank == 0:
                    problem = train_loader.advance_batch()
                dist.monitored_barrier()
                if rank != 0:
                    problem = train_loader.get_problem()
            else:
                problem = train_loader.get_problem()

            if problem is not None:
                agent.model.eval()
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
                    f_traj, b_traj = traj
                    sol_len = len(f_traj)
                else:
                    f_traj = b_traj = None
                    sol_len = np.nan

                res = Result(
                    id=problem.id,
                    time=end_time - start_time,
                    fexp=n_forw_expanded,
                    bexp=n_backw_expanded,
                    len=sol_len,
                    f_traj=f_traj,
                    b_traj=b_traj,
                )
                results_queue.put(res)

            else:  # end batch
                done_batch = True
                dist.monitored_barrier()
                if rank == 0:
                    batch_buffer: list[Result] = []
                    while not results_queue.empty():
                        batch_buffer.append(results_queue.get())

                    start_idx = len(results)
                    end_idx = len(batch_buffer)
                    results.append(batch_buffer)
                    batch_df = results[start_idx:end_idx].get_df(
                        agent.has_policy,
                        agent.has_heuristic,
                        agent.is_bidirectional,
                    )
                    print(f"\nBatch {batch}")
                    print(
                        tabulate(
                            batch_df,
                            headers="keys",
                            tablefmt="psql",
                            showindex=False,
                            floatfmt=".2f",
                        )
                    )
                    # update rank 0 model
                    trajs = [
                        (res.f_traj, res.b_traj)
                        for res in batch_buffer
                        if res.f_traj is not None
                    ]
                    random.shuffle(trajs)

                    if len(trajs) > 0:
                        to.set_grad_enabled(True)
                        agent.model.train()
                        for traj in trajs:
                            for grad_step in range(1, args.grad_steps + 1):
                                agent.optimizer.zero_grad(set_to_none=False)
                                loss = agent.loss_fn(traj[0], agent.model)
                                if agent.is_bidirectional:
                                    b_loss = agent.loss_fn(traj[1], agent.model)
                                    loss += b_loss
                                loss.backward()
                                agent.optimizer.step()

                    batch_total_time = timer() - batch_start_time
                    print(f"Batch time: {batch_total_time:.2f}s")

                # sync models
                if rank == 0:
                    ts = timer()
                all_params_list = [
                    param.data.view(-1) for param in agent.model.parameters()
                ]
                all_params = to.cat(all_params_list)
                dist.broadcast(all_params, src=0)
                offset = 0
                for param in agent.model.parameters():
                    numel = param.numel()
                    param.data.copy_(
                        all_params[offset : offset + numel].view_as(param.data)
                    )
                    offset += numel
                # if rank == 0:
                #     ts = timer()
                # for param in agent.model.parameters():
                #     dist.broadcast(param.data, src=0)
                if rank == 0:
                    print(f"Model sync took {timer() - ts:.2f}s")

                if batch * args.batch_size >= len(train_loader):  # end epoch
                    done_epoch = True
                    # log all search results
                    if rank == 0:
                        stage_search_df = results.get_df(
                            agent.has_policy,
                            agent.has_heuristic,
                            agent.is_bidirectional,
                        )
                        # stage_model_train_df = pd.DataFrame(
                        #     {
                        #         "floss": pd.Series(
                        #             flosses, dtype=pd.Float32Dtype(), index=batch
                        #         ),
                        #     },
                        # )

                        agent.save_model("latest", log=False)
                        with (args.logdir / f"search_train_e{epoch}.pkl").open(
                            "wb"
                        ) as f:
                            pickle.dump(stage_search_df, f)
                        # with (args.logdir / f"model_train_e{epoch}.pkl").open(
                        #     "wb"
                        # ) as f:
                        #     pickle.dump(stage_model_train_df, f)

                        print_search_summary(
                            stage_search_df,
                            agent.is_bidirectional,
                        )
                        # print_model_train_summary(
                        #     stage_model_train_df,
                        #     agent.is_bidirectional,
                        #     agent.has_policy,
                        # )
                        del stage_search_df

                # Validation checks
                if done_epoch:
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
                        results_queue,
                        print_results=False,
                        batch=batch,
                    )

                    if rank == 0:
                        (
                            valid_solved,
                            valid_total_expanded,
                            valid_total_time,
                        ) = valid_results

                        if valid_total_expanded <= best_valid_expanded:
                            best_valid_expanded = valid_total_expanded
                            print("Saving best model")
                            best_models_log.write(
                                f"solved {valid_solved} exp {valid_total_expanded}\n"
                            )
                            agent.save_model("best_expanded", log=False)

                        print(f"\nTime: {epoch_total_time:.2f}s")
                        print(f"\nEND EPOCH {epoch}")
                        print(
                            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                        )
                        sys.stdout.flush()
                        epoch_total_time = timer() - epoch_start_time
                        simple_log.write(f"{epoch} {epoch_total_time:.2f}\n")
                        simple_log.flush()
                        epoch += 1

                # Checkpoint at beginning of batch b
                if (batch % checkpoint_every_n_batches == 0) or done_epoch:
                    ts = timer()
                    loader_state = None
                    # dist.gather_object(
                    #     train_loader.get_state(), loader_states if rank == 0 else None, dst=0
                    # )

                    if rank == 0:
                        checkpoint_data = {
                            "search_results": results.results,
                            "model_state": agent.model.state_dict(),
                            "optimizer_state": agent.optimizer.state_dict(),
                            "expansion_budget": expansion_budget,
                            "best_valid_expanded": best_valid_expanded,
                            "time_in_epoch": timer() - epoch_start_time,
                            "time_in_training": timer() - train_start_time,
                            "epoch": epoch,
                            "batch": batch,
                            "done_epoch": done_epoch,
                            "loader_state": loader_state,
                        }
                        new_checkpoint_path = args.logdir / f"checkpoint_b{batch}.pkl"
                        with new_checkpoint_path.open("wb") as f:
                            to.save(checkpoint_data, f)
                        del loader_state, checkpoint_data
                        if not args.keep_all_checkpoints:
                            old_checkpoint_path.unlink(missing_ok=True)
                            old_checkpoint_path = new_checkpoint_path
                        print(
                            f"\nCheckpoint saved to {new_checkpoint_path.name}, took {timer() - ts:.2f}s"
                        )

    dist.monitored_barrier()  # done training
