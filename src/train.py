import math
import os
from pathlib import Path
import random
import time
from typing import Callable, Type, Union

import numpy as np
import pandas as pd
from tabulate import tabulate
import torch as to
import torch.distributed as dist
from torch.utils.tensorboard.writer import SummaryWriter
import tqdm

from domains.domain import Domain, Problem
from search import MergedTrajectory


def train(
    rank,
    agent,
    model: Union[to.nn.Module, tuple[to.nn.Module, to.nn.Module]],
    model_save_path: Path,
    loss_fn: Callable,
    optimizer_cons: Type[to.optim.Optimizer],
    optimizer_params: dict,
    problems: list[Problem],
    local_batch_size: int,
    writer: SummaryWriter,
    world_size: int,
    update_levin_costs: bool,
    initial_budget: int,
    seed: int,
    grad_steps: int = 10,
    shuffle_trajectory=False,
):
    current_budget = initial_budget
    dummy_last = False

    if world_size > 1:
        sh_t = to.zeros(1, dtype=to.int64) + len(problems)
        dist.all_reduce(sh_t, dist.ReduceOp.SUM)
        world_num_problems = int(sh_t[0].item())

        n_batches = math.ceil(len(problems) / local_batch_size)
        sh_t[0] = n_batches
        dist.all_reduce(sh_t, dist.ReduceOp.MAX)
        if n_batches < sh_t[0]:
            dummy_last = True
    else:
        world_num_problems = len(problems)

    world_batches_per_epoch = math.ceil(
        world_num_problems / (local_batch_size * world_size)
    )

    # try to log at most 10k histograms per param, assuming an upper bound of 100 epochs
    param_log_interval = max(1, int(world_batches_per_epoch / 200))

    search_result_header = [
        "ProblemId",
        "SolutionLength",
        "NumExpanded",
        "NumGenerated",
        "Time",
    ]

    dummy_data = np.column_stack(
        (
            range(world_num_problems),
            np.zeros(
                (world_num_problems, len(search_result_header) - 1), dtype=np.int64
            ),
        )
    )
    world_results_df = pd.DataFrame(dummy_data, columns=search_result_header)
    del dummy_data
    world_results_df["Time"] = world_results_df["Time"].astype(float, copy=False)
    world_results_df.set_index("ProblemId", inplace=True)

    opt_result_header = f"OptStep   Loss    Acc"

    bidirectional = agent.bidirectional
    if bidirectional:
        assert isinstance(model, tuple)
        f_model, b_model = model
        f_model_save_path = model_save_path / "forward.pt"
        b_model_save_path = model_save_path / "backward.pt"

        forward_optimizer = optimizer_cons(f_model.parameters(), **optimizer_params)
        backward_optimizer = optimizer_cons(b_model.parameters(), **optimizer_params)

        for param in b_model.parameters():
            if not param.grad:
                param.grad = to.zeros_like(param)
    else:
        assert isinstance(model, to.nn.Module)
        f_model = model
        f_model_save_path = model_save_path / "forward.pt"
        forward_optimizer = optimizer_cons(f_model.parameters(), **optimizer_params)

    for param in f_model.parameters():
        if not param.grad:
            param.grad = to.zeros_like(param)

    if rank == 0:
        log_params(writer, f_model, "forward", 0)
        if bidirectional:
            log_params(writer, b_model, "backward", 0)

    problems_loader = ProblemsBatchLoader(
        problems,
        batch_size=local_batch_size,
        seed=seed,
        shuffle=True,
        dummy_last=dummy_last,
    )

    local_batch_opt_results = to.zeros(3, dtype=to.float64)
    world_batch_results = [
        to.zeros((local_batch_size, 5), dtype=to.int64) for _ in range(world_size)
    ]

    batches_seen = 0
    epoch = 0
    solved_problems = set()
    total_num_expanded = 0
    forward_opt_steps = 0
    backward_opt_steps = 0

    forward_trajs = {"one_state": []}
    backward_trajs = {"one_state": []}

    world_epoch_f_loss = np.zeros(world_batches_per_epoch)
    world_epoch_f_acc = np.zeros(world_batches_per_epoch)
    world_epoch_b_loss = np.zeros(world_batches_per_epoch)
    world_epoch_b_acc = np.zeros(world_batches_per_epoch)

    while len(solved_problems) < world_num_problems:
        epoch += 1

        num_new_problems_solved_this_epoch = 0
        num_problems_solved_this_epoch = 0

        if rank == 0:
            print(
                "============================================================================"
            )
            print(f"START EPOCH {epoch} BUDGET {current_budget}")
            print(
                "============================================================================\n"
            )

        if rank == 0:
            problems_loader = tqdm.tqdm(problems_loader, total=world_batches_per_epoch)

        for local_batch_problems in problems_loader:
            batches_seen += 1

            if rank == 0:
                print(f"\n\nBatch {batches_seen}")

            to.set_grad_enabled(False)

            forward_trajs["current"] = []
            f_model.eval()
            backward_trajs["current"] = []
            if bidirectional:
                b_model.eval()  # type:ignore

            num_problems_solved_this_batch = 0
            local_batch_search_results = to.zeros(local_batch_size, 5, dtype=to.int64)
            for i, problem in enumerate(local_batch_problems):
                start_time = time.time()
                (solution_length, num_expanded, num_generated, traj,) = agent.search(
                    problem,
                    model,
                    current_budget,
                    update_levin_costs,
                    train=True,
                )
                end_time = time.time()
                if bidirectional:
                    problem.domain.reset()

                # todo handle single state trajectories, they break batchnorm layers
                local_batch_search_results[i, 0] = problem[0]
                local_batch_search_results[i, 1] = solution_length
                local_batch_search_results[i, 2] = num_expanded
                local_batch_search_results[i, 3] = num_generated
                local_batch_search_results[i, 4] = int((end_time - start_time) * 1000)

                if traj:
                    forward_trajs["current"].append(traj[0])
                    if bidirectional:
                        backward_trajs["current"].append(traj[1])

            if world_size > 1:
                dist.all_gather(world_batch_results, local_batch_search_results)
                world_batch_results_t = to.cat(world_batch_results, dim=0)
            else:
                world_batch_results_t = local_batch_search_results

            world_batch_results_arr = world_batch_results_t.numpy()
            # hacky way to filter out results from partial batches
            world_batch_results_arr = world_batch_results_arr[
                world_batch_results_arr[:, 2] > 0
            ]

            world_batch_ids = world_batch_results_arr[:, 0]
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
                trajs_dict: dict,
                opt_step: int,
                name: str,
            ):
                trajs = trajs_dict["current"]
                if len(trajs_dict["one_state"]) == 1:
                    trajs.append(trajs_dict["one_state"].pop())
                if len(trajs) == 1 and len(trajs[0]) == 1:
                    trajs_dict["one_state"].append(trajs.pop())

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

                    if world_size > 1:
                        dist.all_reduce(local_batch_opt_results, op=dist.ReduceOp.SUM)
                        num_procs_found_solution = int(
                            local_batch_opt_results[2].item()
                        )
                        if num_procs_found_solution > 0:
                            sync_grads(model, num_procs_found_solution)

                    num_procs_found_solution = int(local_batch_opt_results[2].item())

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
                        # print(local_batch_opt_results)
                        opt_step += 1
                        if rank == 0:
                            opt_passes = opt_step // grad_steps
                            step_within_opt_pass = opt_step % grad_steps
                            if step_within_opt_pass == 1 or step_within_opt_pass == 0:
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

            idx = (batches_seen % world_batches_per_epoch) - 1

            forward_opt_steps, f_loss, f_acc = fit_model(
                f_model,
                forward_optimizer,
                forward_trajs,
                forward_opt_steps,
                name="forward",
            )
            world_epoch_f_loss[idx] = f_loss
            world_epoch_f_acc[idx] = f_acc

            if bidirectional:
                backward_opt_steps, b_loss, b_acc = fit_model(
                    b_model,  # type:ignore
                    backward_optimizer,  # type:ignore
                    backward_trajs,  # type:ignore
                    backward_opt_steps,  # type:ignore
                    name="backward",
                )
                world_epoch_b_loss[idx] = b_loss
                world_epoch_b_acc[idx] = b_acc

            if rank == 0:
                if (
                    batches_seen % param_log_interval == 0
                    or len(solved_problems) == world_num_problems
                ):
                    to.save(f_model.state_dict(), f_model_save_path)
                    log_params(writer, f_model, "forward", batches_seen)
                    if bidirectional:
                        to.save(b_model.state_dict(), b_model_save_path)
                        log_params(writer, b_model, "backward", batches_seen)

                batch_avg = num_problems_solved_this_batch / num_problems_this_batch
                # fmt: off
                writer.add_scalar(f"budget_{current_budget}/solved_vs_batch", batch_avg, batches_seen)
                writer.add_scalar(f"cum_unique_solved_vs_batch", len(solved_problems), batches_seen)
                # fmt: on

        if rank == 0:
            epoch_solved = num_problems_solved_this_epoch / world_num_problems
            epoch_f_loss = world_epoch_f_loss.mean(where=(world_epoch_f_loss >= 0))
            epoch_f_acc = world_epoch_f_acc.mean(where=(world_epoch_f_acc >= 0))
            if bidirectional:
                epoch_b_loss = world_epoch_b_loss.mean(where=(world_epoch_b_loss >= 0))
                epoch_b_acc = world_epoch_b_acc.mean(where=(world_epoch_b_acc >= 0))
            print(
                "============================================================================"
            )
            print(f"END EPOCH {epoch} BUDGET {current_budget}")
            print(
                f"CURRENT {num_problems_solved_this_epoch}/{world_num_problems}\n"
                f"OVERALL {len(solved_problems)}/{world_num_problems}, +{num_new_problems_solved_this_epoch}, {world_num_problems - len(solved_problems)} remaining\n"
                f"Average forward loss: {epoch_f_loss:5.3f}, acc: {epoch_f_acc:5.3f}"
            )
            if bidirectional:
                print(
                    f"Average backward loss: {epoch_b_loss:5.3f}, acc: {epoch_b_acc:5.3f}"
                )
            print(
                "============================================================================\n"
            )

            # fmt: off
            writer.add_scalar("budget_vs_epoch", current_budget, epoch)
            writer.add_scalar(f"budget_{current_budget}/solved_vs_epoch", epoch_solved, epoch)
            writer.add_scalar("cum_unique_solved_vs_epoch", len(solved_problems), epoch)

            writer.add_scalar(f"loss_vs_epoch/forward", epoch_f_loss, epoch)
            writer.add_scalar(f"acc_vs_epoch/forward", epoch_f_acc, epoch)
            if bidirectional:
                writer.add_scalar(f"loss_vs_epoch/backward", epoch_b_loss, epoch)
                writer.add_scalar(f"acc_vs_epoch/backward", epoch_b_acc, epoch)

            world_results_df.to_csv(f"{writer.log_dir}/epoch_{epoch}.csv")
            # fmt: on

        if num_new_problems_solved_this_epoch == 0:
            current_budget *= 2


class ProblemsBatchLoader:
    def __init__(
        self,
        problems: list[Problem],
        batch_size: int,
        seed: int = 1,
        shuffle: bool = True,
        dummy_last: bool = False,
    ):
        self.rng = np.random.default_rng(seed)
        self.problems = np.empty(len(problems), dtype=object)
        self.problems[:] = problems
        self.batch_size = batch_size
        self._len = len(problems)
        self._num_problems_served = 0
        self._shuffle = shuffle

        self.dummy_last = dummy_last
        self._dummy_served = False

    def __len__(self):
        return self._len

    def __iter__(self):
        self._dummy_served = False
        if self._shuffle:
            self._indices = self.rng.permutation(self._len)
        else:
            self._indices = np.arange(self._len)

        self._num_problems_served = 0

        return self

    def __next__(self):
        if self._num_problems_served >= self._len:
            if self.dummy_last and not self._dummy_served:
                self._dummy_served = True
                return []

            raise StopIteration
        next_indices = self._indices[
            self._num_problems_served : self._num_problems_served + self.batch_size
        ]
        self._num_problems_served += len(next_indices)
        return self.problems[next_indices]

    def __getitem__(self, idx):
        return self.problems[idx]


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
