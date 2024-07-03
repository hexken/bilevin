from argparse import Namespace
import pickle
import random
import sys
from timeit import default_timer as timer

import numpy as np
from tabulate import tabulate
import torch as to
import torch.distributed as dist
import torch.multiprocessing as mp

from search.agent import Agent
from search.loaders import AsyncProblemLoader
from search.utils import Result, ResultsLog, print_search_summary
from test import test


def train(
    args: Namespace,
    rank: int,
    agent: Agent,
    train_loader: AsyncProblemLoader,
    valid_loader: AsyncProblemLoader,
    results_queue: mp.Queue,
):

    if rank == 0:
        simple_log = (args.logdir / "simple_log.txt").open("a")

    best_valid_expanded = len(valid_loader) * args.train_expansion_budget + 1

    if args.checkpoint_path is None:
        results = ResultsLog(None, agent)
        epoch = 0
        done_epoch = False
        train_start_time = timer()
    else:
        with args.checkpoint_path.open("rb") as f:
            checkpoint_data = to.load(f)
            agent.optimizer.load_state_dict(checkpoint_data["optimizer_state"])
            results = ResultsLog(checkpoint_data["search_results"], agent)
            best_valid_expanded = checkpoint_data["best_valid_expanded"]
            epoch_start_time = timer() - checkpoint_data["time_in_epoch"]
            train_start_time = timer() - checkpoint_data["time_in_training"]
            epoch = checkpoint_data["epoch"]
            batch = checkpoint_data["batch"]
            done_epoch = checkpoint_data["done_epoch"]
            train_loader.load_state_dict(checkpoint_data["loader_state"])

        if rank == 0:
            print(
                "----------------------------------------------------------------------------"
            )
            print(f"Continuing epoch {epoch} using checkpoint {args.checkpoint_path}")
            print(
                "----------------------------------------------------------------------------"
            )

    old_checkpoint_path = args.logdir / f"dummy_chkpt"

    dist.monitored_barrier()
    done_batch = True
    while True:
        if done_batch:
            batch_start_time = timer()
            done_batch = False
            if done_epoch or epoch == 0:
                if epoch == args.n_epochs:
                    break
                epoch_start_time = timer()
                done_epoch = False
                results = ResultsLog(None, agent)
                epoch += 1
                batch = 0
                if rank == 0:
                    train_loader.init_indexer(shuffle=args.shuffle)
                    print(
                        "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                    )
                    print(f"START EPOCH {epoch}")
            batch += 1
            if rank == 0:
                problem = train_loader.advance_batch()
            dist.monitored_barrier()
            if rank != 0:
                problem = train_loader.get_problem()
        else:
            problem = train_loader.get_problem()

        if problem is not None:
            # print(f"rank {rank} problem {problem.id}")
            agent.model.eval()
            to.set_grad_enabled(False)

            start_time = timer()
            (
                n_forw_expanded,
                n_backw_expanded,
                (f_traj, b_traj),
            ) = agent.search(
                problem,
                args.train_expansion_budget,
                time_budget=args.time_budget,
            )
            end_time = timer()

            if f_traj is not None:
                sol_len = len(f_traj)
            else:
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

                results.append(batch_buffer)
                batch_df = results[-len(batch_buffer) :].get_df()
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
                train_loader.rng.shuffle(trajs)

                if len(trajs) > 0:
                    to.set_grad_enabled(True)
                    agent.model.train()
                    for grad_step in range(1, args.grad_steps + 1):
                        f_losses = []
                        b_losses = []
                        for f_traj, b_traj in trajs:
                            agent.optimizer.zero_grad(set_to_none=False)
                            loss = agent.loss_fn(f_traj, agent.model)
                            f_losses.append(loss.item())
                            if agent.is_bidirectional:
                                b_loss = agent.loss_fn(b_traj, agent.model)
                                b_losses.append(b_loss.item())
                                loss += b_loss
                            loss.backward()
                            agent.optimizer.step()
                        if agent.is_bidirectional:
                            print(f"{np.mean(f_losses):.2f}\t{np.mean(b_losses):.2f}")
                        else:
                            print(f"{np.mean(f_losses):.2f}")

                batch_total_time = timer() - batch_start_time
                print(
                    f"Batch time: {batch_total_time:.2f}s, solved {len(trajs)}/{len(batch_buffer)}"
                )
                print(
                    f"Epoch time: {timer() - epoch_start_time:.2f}s, solved {results.solved}/{len(train_loader)}"
                )
                sys.stdout.flush()

            # sync models
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

            if batch * train_loader.batch_size >= len(train_loader):
                done_epoch = True
                # log all search results
                if rank == 0:
                    results_df = results.get_df()

                    agent.save_model("final", log=False)
                    with (args.logdir / f"train_e{epoch}.pkl").open("wb") as f:
                        pickle.dump(results_df, f)

                    print()
                    epoch_solved, epoch_exp = print_search_summary(
                        results_df,
                        agent.is_bidirectional,
                    )

            # Validation checks
            if done_epoch:
                if rank == 0:
                    print("\nValidating...\n")
                    valid_start_time = timer()
                    sys.stdout.flush()

                dist.monitored_barrier()

                valid_df = test(
                    args,
                    rank,
                    agent,
                    valid_loader,
                    results_queue,
                    print_results=False,
                )

                if rank == 0:
                    valid_exp = valid_df["exp"].sum()
                    valid_solved = len(valid_df[valid_df["len"] > 0])
                    with (args.logdir / f"valid_e{epoch}.pkl").open("wb") as f:
                        pickle.dump(valid_df, f)
                    print_search_summary(valid_df, agent.is_bidirectional)

                    if valid_exp <= best_valid_expanded:
                        best_valid_expanded = valid_exp
                        print("Saving best model")
                        agent.save_model("best_expanded", log=False)

                    print(f"\nValidation time: {timer() - valid_start_time:.2f}s")
                    print(f"Epoch time: {timer() -epoch_start_time:.2f}s")
                    print(f"\nEND EPOCH {epoch}")
                    print(
                        ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                    )
                    sys.stdout.flush()
                    simple_log.write(
                        f"{epoch} {epoch_solved} {epoch_exp} {valid_solved} {valid_exp} {timer() - train_start_time:.2f}\n"
                    )
                    simple_log.flush()
                    del results_df, valid_df

            # Checkpoint at beginning of batch b
            if (batch % args.checkpoint_every_n_batch == 0) or done_epoch:
                ts = timer()
                if rank == 0:
                    checkpoint_data = {
                        "search_results": results.results,
                        "model_state": agent.model.state_dict(),
                        "optimizer_state": agent.optimizer.state_dict(),
                        "expansion_budget": args.train_expansion_budget,
                        "best_valid_expanded": best_valid_expanded,
                        "time_in_epoch": timer() - epoch_start_time,
                        "time_in_training": timer() - train_start_time,
                        "epoch": epoch,
                        "batch": batch,
                        "done_epoch": done_epoch,
                        "loader_state": train_loader.state_dict(),
                    }
                    new_checkpoint_path = (
                        args.logdir / f"checkpoint_e{epoch}b{batch}.pkl"
                    )
                    with new_checkpoint_path.open("wb") as f:
                        to.save(checkpoint_data, f)
                    del checkpoint_data
                    if not args.keep_all_checkpoints:
                        old_checkpoint_path.unlink(missing_ok=True)
                        old_checkpoint_path = new_checkpoint_path
                    print(
                        f"\nCheckpoint saved to {new_checkpoint_path.name}, took {timer() - ts:.2f}s"
                    )
    dist.monitored_barrier()  # done training
