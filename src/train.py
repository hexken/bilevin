import math
from pathlib import Path
import time
from typing import Type, Callable, Union

import numpy as np
import pandas as pd
from domains.domain import Domain
from tabulate import tabulate
import torch as to
import torch.distributed as dist
from torch.utils.tensorboard.writer import SummaryWriter
import tqdm

from search import MergedTrajectory


def train(
    agent,
    model: Union[to.nn.Module, tuple[to.nn.Module, to.nn.Module]],
    model_path: Path,
    loss_fn: Callable,
    optimizer_cons: Type[to.optim.Optimizer],
    optimizer_params: dict,
    problems: list[tuple[int, Domain]],
    local_batch_size: int,
    writer: SummaryWriter,
    world_size: int,
    initial_budget: int,
    grad_steps: int = 10,
    shuffle_trajectory=False,
    track_params: bool = False,
):
    current_budget = initial_budget

    if world_size > 1:
        rank = dist.get_rank()
    else:
        rank = 0

    world_num_problems = len(problems) * world_size
    world_batch_size = local_batch_size * world_size
    world_batches_per_epoch = math.ceil(
        world_num_problems / (local_batch_size * world_size)
    )

    # try to log at most 10k histograms per param, assuming an upper bound of 100 epochs
    if track_params:
        param_log_interval = max(1, int(world_batches_per_epoch / 100))

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
                (world_num_problems, len(search_result_header) - 1), dtype=np.int32
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
        forward_model, backward_model = model

        forward_optimizer = optimizer_cons(
            forward_model.parameters(), **optimizer_params
        )
        forward_model_path = model_path

        backward_optimizer = optimizer_cons(
            backward_model.parameters(), **optimizer_params
        )
        backward_model_path = Path(
            str(model_path).replace("_forward.pt", "_backward.pt")
        )

        for param in backward_model.parameters():
            param.grad = to.zeros_like(param)
    else:
        assert isinstance(model, to.nn.Module)
        forward_model = model
        forward_model_path = model_path
        forward_optimizer = optimizer_cons(
            forward_model.parameters(), **optimizer_params
        )

    for param in forward_model.parameters():
        param.grad = to.zeros_like(param)

    problems_loader = ProblemsBatchLoader(
        problems, batch_size=local_batch_size, shuffle=True
    )

    local_batch_opt_results = to.zeros(3, dtype=to.float32)
    local_batch_search_results = to.zeros(local_batch_size, 5, dtype=to.int32)
    world_batch_results = [
        to.zeros((local_batch_size, 5), dtype=to.int32) for _ in range(world_size)
    ]

    batches_seen = 0
    epoch = 0

    solved_problems = set()

    forward_opt_steps = 0
    backward_opt_steps = 0

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
            print(f"\nBeginning epoch {epoch} with budget {current_budget}")
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

            forward_trajs = []
            forward_model.eval()
            backward_trajs = []
            if bidirectional:
                backward_model.eval()  # type:ignore

            num_problems_solved_this_batch = 0
            for i, problem in enumerate(local_batch_problems):
                start_time = time.time()
                (solution_length, num_expanded, num_generated, traj,) = agent.search(
                    problem,
                    model,
                    current_budget,
                    train=True,
                )
                end_time = time.time()

                local_batch_search_results[i, 0] = problem[0]
                local_batch_search_results[i, 1] = solution_length
                local_batch_search_results[i, 2] = num_expanded
                local_batch_search_results[i, 3] = num_generated
                local_batch_search_results[i, 4] = int((end_time - start_time) * 1000)

                if traj:
                    forward_trajs.append(traj[0])
                    if bidirectional:
                        backward_trajs.append(traj[1])

            if world_size > 1:
                dist.all_gather(world_batch_results, local_batch_search_results)
                world_batch_results_t = to.cat(world_batch_results, dim=0)
            else:
                world_batch_results_t = local_batch_search_results

            world_batch_results_arr = world_batch_results_t.numpy()

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

            if rank == 0:
                print(
                    tabulate(
                        world_batch_results_df,
                        headers="keys",
                        tablefmt="psql",
                    )
                )
                print(f"Solved {num_problems_solved_this_batch}/{world_batch_size}\n")

            def fit_model(
                model: to.nn.Module,
                optimizer: to.optim.Optimizer,
                trajs: list,
                opt_step: int,
                name: str,
            ):
                if rank == 0 and trajs:
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

                    if world_size > 1:
                        sync_grads(model, world_size)

                    # todo grad clipping? for now inspect norms
                    if trajs and rank == 0:
                        total_norm = 0
                        for p in model.parameters():
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                        total_norm = total_norm**0.5
                        # print(total_norm)
                        writer.add_scalar(
                            f"total_grad_norm/{name}", total_norm, opt_step
                        )

                    optimizer.step()

                    if trajs:
                        avg_action_nll = avg_action_nll.item()
                        with to.no_grad():
                            acc = (
                                (logits.argmax(dim=1) == merged_traj.actions).sum()
                                / len(logits)
                            ).item()

                        local_batch_opt_results[0] = avg_action_nll
                        local_batch_opt_results[1] = acc
                        local_batch_opt_results[2] = 1
                    else:
                        local_batch_opt_results[:] = 0

                    if world_size > 1:
                        dist.all_reduce(local_batch_opt_results, op=dist.ReduceOp.SUM)

                    num_procs_found_solution = local_batch_opt_results[2].item()
                    if num_procs_found_solution > 0:
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
                                    writer.add_scalar( f"loss_vs_opt_step/{name}", loss, opt_passes,)
                                    writer.add_scalar( f"acc_vs_opt_step/{name}", acc, opt_passes,)
                                    # fmt:on

                if rank == 0 and num_procs_found_solution > 0:
                    if track_params and (
                        batches_seen % param_log_interval == 0
                        or len(solved_problems) == world_num_problems
                    ):
                        for (
                            param_name,
                            param,
                        ) in forward_model.named_parameters():
                            writer.add_histogram(
                                f"param_vs_opt_pass/{name}/{param_name}",
                                param.data,
                                opt_passes,
                                bins=512,
                            )
                    print("")
                return opt_step, loss, acc

            idx = (batches_seen % world_batches_per_epoch) - 1

            forward_opt_steps, f_loss, f_acc = fit_model(
                forward_model,
                forward_optimizer,
                forward_trajs,
                forward_opt_steps,
                name="forward",
            )
            world_epoch_f_loss[idx] = f_loss
            world_epoch_f_acc[idx] = f_acc

            if bidirectional:
                backward_opt_steps, b_loss, b_acc = fit_model(
                    backward_model,  # type:ignore
                    backward_optimizer,  # type:ignore
                    backward_trajs,  # type:ignore
                    backward_opt_steps,  # type:ignore
                    name="backward",
                )
                world_epoch_b_loss[idx] = b_loss
                world_epoch_b_acc[idx] = b_acc

            if rank == 0:
                if (
                    batches_seen % 100 == 0
                    or len(solved_problems) == world_num_problems
                ):
                    to.save(forward_model, forward_model_path)  # type:ignore
                    if bidirectional:
                        to.save(backward_model, backward_model_path)  # type:ignore
                batch_avg = num_problems_solved_this_batch / world_batch_size
                # fmt: off
                writer.add_scalar(f"budget_{current_budget}/solved_vs_batch", batch_avg, batches_seen)
                # writer.add_scalar(f"cum_unique_solved_vs_batch", len(solved_problems), total_batches)
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
            print(
                f"Completed epoch {epoch}, solved {num_problems_solved_this_epoch}/{world_num_problems} problems with budget {current_budget}\n"
                f"Solved {num_new_problems_solved_this_epoch} new problems, {world_num_problems - len(solved_problems)} remaining\n"
                f"Average forward epoch loss: {epoch_f_loss}, acc: {epoch_f_acc}"
            )
            if bidirectional:
                print(
                    f"Average backward epoch loss: {epoch_b_loss}, acc: {epoch_b_acc}"
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

            writer.add_text(f"epoch{epoch}_results", world_results_df.to_markdown(), epoch)
            # fmt: on

        if num_new_problems_solved_this_epoch == 0:
            current_budget *= 2


class ProblemsBatchLoader:
    def __init__(
        self, problems: list[tuple[int, Domain]], batch_size: int, shuffle: bool = True
    ):
        self.problems = np.array(problems)
        self.batch_size = batch_size
        self._len = len(problems)
        self._num_problems_served = 0
        self._shuffle = shuffle

    def __len__(self):
        return self._len

    def __iter__(self):
        if self._shuffle:
            self._indices = np.random.permutation(self._len)
        else:
            self._indices = np.arange(self._len)

        self._num_problems_served = 0

        return self

    def __next__(self):
        if self._num_problems_served >= self._len:
            raise StopIteration
        next_indices = self._indices[
            self._num_problems_served : self._num_problems_served + self.batch_size
        ]
        self._num_problems_served += len(next_indices)
        return self.problems[next_indices]

    def __getitem__(self, idx):
        return self.problems[idx]


def sync_grads(model: to.nn.Module, world_size: int):
    # assumes grads are not None
    # todo scale gradients properly since not all processes contribute equally
    all_grads_list = []
    for param in model.parameters():
        assert param.grad is not None
        all_grads_list.append(param.grad.view(-1))
    all_grads = to.cat(all_grads_list)
    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
    offset = 0
    for param in model.parameters():
        assert param.grad is not None
        param.grad.data.copy_(
            all_grads[offset : offset + param.numel()].view_as(param.grad.data)
        )
        offset += param.numel()
