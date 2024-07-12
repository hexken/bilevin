from argparse import Namespace
import gc
import pickle
from queue import Empty
import sys
from timeit import default_timer as timer

import numpy as np
from tabulate import tabulate
import torch as to
import torch.distributed as dist
import torch.multiprocessing as mp

from search.agent import Agent
from search.loaders import ArrayLoader
from search.utils import Result, ResultsLog, print_search_summary
from test import test
# import objgraph
from pympler import muppy, summary, tracker


def train(
    args: Namespace,
    rank: int,
    agent: Agent,
    train_loader: ArrayLoader,
    valid_loader: ArrayLoader,
    results_queue: mp.Queue,
):

    if rank == 0:
        simple_log = (args.logdir / "simple_log.txt").open("a")
        old_checkpoint_path = args.logdir / f"dummy_chkpt"
        best_valid_expanded = len(valid_loader) * args.train_expansion_budget + 1
        batch_buffer: list[Result] = []

    results = ResultsLog(agent)
    if args.checkpoint_path is None:
        epoch = 0
        done_epoch = False
        train_start_time = timer()
    else:
        with args.checkpoint_path.open("rb") as f:
            checkpoint_data = to.load(f)
            results.load_state_dict(checkpoint_data["results_state"])
            agent.model.load_state_dict(checkpoint_data["model_state"])
            agent.optimizer.load_state_dict(checkpoint_data["optimizer_state"])
            best_valid_expanded = checkpoint_data["best_valid_expanded"]
            epoch_start_time = timer() - checkpoint_data["time_in_epoch"]
            train_start_time = timer() - checkpoint_data["time_in_training"]
            epoch = checkpoint_data["epoch"]
            batch = checkpoint_data["batch"]
            done_epoch = checkpoint_data["done_epoch"]
            train_loader.load_state_dict(checkpoint_data["loader_state"])
            del checkpoint_data

        if rank == 0:
            print(
                "----------------------------------------------------------------------------"
            )
            print(f"Continuing epoch {epoch} using checkpoint {args.checkpoint_path}")
            print(
                "----------------------------------------------------------------------------"
            )

    done_batch = True
    dist.barrier()
    while True:
        if done_batch:
            dist.barrier()
            batch_start_time = timer()
            done_batch = False
            if done_epoch or epoch == 0:
                if epoch == args.n_epochs:
                    break
                epoch_start_time = timer()
                done_epoch = False
                results.clear()
                epoch += 1
                batch = 0
                if rank == 0:
                    train_loader.reset_indices(shuffle=args.shuffle)

                    print(
                        "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                    )
                    print(f"START EPOCH {epoch}")
            batch += 1
            if rank == 0:
                train_loader.next_batch()
            gc.collect()
            dist.barrier()
            if rank == 0:
                all_objects = muppy.get_objects()
                sum1 = summary.summarize(all_objects)
                summary.print_(sum1)
                sys.stdout.flush()
            dist.barrier()

        problem = train_loader.get()
        if problem is not None:
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
            if end_time - start_time > args.slow_problem:
                print(
                    f"Rank {rank} problem {problem.id} SLOW {end_time - start_time:.2f}s"
                )
                sys.stdout.flush()

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
            del res, f_traj, b_traj, problem

        else:  # end batch
            done_batch = True
            dist.barrier()
            if rank == 0:
                print(f"\nBatch {batch}")
                for _ in range(train_loader.current_batch_size(batch)):
                    res = results_queue.get()
                    batch_buffer.append(res)
                    del res

                results.append(batch_buffer)
                batch_df = results[-len(batch_buffer) :].get_df(sort=True)
                print(
                    tabulate(
                        batch_df,
                        headers="keys",
                        tablefmt="psql",
                        showindex=False,
                        floatfmt=".2f",
                    )
                )
                del batch_df
                # update rank 0 model
                trajs = [
                    (res.id, res.f_traj, res.b_traj)
                    for res in batch_buffer
                    if res.f_traj is not None
                ]

                # to make the shuffle deterministic
                trajs = sorted(trajs, key=lambda x: x[0])
                train_loader.rng.shuffle(trajs)

                if len(trajs) > 0:
                    print("Updating model...")
                    to.set_grad_enabled(True)
                    agent.model.train()
                    for _ in range(args.grad_steps):
                        f_losses = []
                        b_losses = []
                        for _, f_traj, b_traj in trajs:
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
                    f"Epoch time: {timer() - epoch_start_time:.2f}s, solved {results.solved}/{len(results)}"
                )
                del trajs
                sys.stdout.flush()
                batch_buffer.clear()

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
            del all_params, all_params_list

            if batch * train_loader.batch_size >= len(train_loader):  # end epoch
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

            # validate
            if done_epoch:
                if rank == 0:
                    print("\nValidating...\n")
                    valid_start_time = timer()
                    sys.stdout.flush()

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

            if (batch % args.checkpoint_every_n_batch == 0) or done_epoch:
                if rank == 0:
                    ts = timer()
                    checkpoint_data = {
                        "results_state": results.state_dict(),
                        "model_state": agent.model.state_dict(),
                        "optimizer_state": agent.optimizer.state_dict(),
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
                    old_checkpoint_path.unlink(missing_ok=True)
                    old_checkpoint_path = new_checkpoint_path
                    print(
                        f"\nCheckpoint saved to {new_checkpoint_path.name}, took {timer() - ts:.2f}s"
                    )
    dist.barrier()
    if rank == 0:
        print(f"\nTraining completed in {timer() - train_start_time:.2f}s")
