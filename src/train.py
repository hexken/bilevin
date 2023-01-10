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
    writer: SummaryWriter,
    world_size: int = 1,
    initial_budget: int = 7000,
    grad_steps: int = 10,
    shuffle_trajectory=False,
    batch_size: int = 32,
    track_params: bool = False,
):
    current_budget = initial_budget

    if world_size > 1:
        rank = dist.get_rank()
    else:
        rank = 0

    num_problems = len(problems) * world_size

    search_result_header = [
        "ProblemId",
        "SolutionLength",
        "NumExpanded",
        "NumGenerated",
        "Time",
    ]
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

    device = next(forward_model.parameters()).device

    local_batch_size = batch_size // world_size
    problems_loader = ProblemsBatchLoader(
        problems, batch_size=local_batch_size, shuffle=True
    )

    local_opt_results = to.zeros(5, dtype=to.float32, device=device)
    local_batch_results = to.zeros(local_batch_size, 5, dtype=to.int32, device=device)
    batch_results = [
        to.zeros((local_batch_size, 5), dtype=to.int32, device=device)
        for _ in range(world_size)
    ]

    total_batches = 0
    epoch = 0

    solved_problems = set()

    forward_opt_steps = 0
    backward_opt_steps = 0
    while len(solved_problems) < num_problems:
        epoch += 1
        if rank == 0:
            print(
                "============================================================================"
            )
            print(f"\nBeginning epoch {epoch}, budget {current_budget}")
            print(
                "============================================================================\n"
            )
        num_new_problems_solved_this_epoch = 0
        num_problems_solved_this_epoch = 0

        if rank == 0:
            problems_loader = tqdm.tqdm(
                problems_loader, total=math.ceil(num_problems / batch_size)
            )

        for batch_problems in problems_loader:
            total_batches += 1

            if rank == 0:
                print(f"\n\nBatch {total_batches}\n")

            forward_trajs = []
            to.set_grad_enabled(False)
            forward_model.eval()
            backward_trajs = []
            if bidirectional:
                backward_model.eval()  # type:ignore

            num_problems_solved_this_batch = 0
            for i, problem in enumerate(batch_problems):
                start_time = time.time()
                (solution_length, num_expanded, num_generated, traj,) = agent.search(
                    problem,
                    model,
                    current_budget,
                    train=True,
                )

                local_batch_results[i, 0] = problem[0]
                local_batch_results[i, 1] = solution_length
                local_batch_results[i, 2] = num_expanded
                local_batch_results[i, 3] = num_generated
                local_batch_results[i, 4] = int((time.time() - start_time) * 1000)

                if solution_length:
                    forward_trajs.append(traj[0])
                    if bidirectional:
                        backward_trajs.append(traj[1])

            if world_size > 1:
                dist.all_gather(batch_results, local_batch_results)
                batch_results_t = to.cat(batch_results, dim=0)
            else:
                batch_results_t = local_batch_results

            batch_df = pd.DataFrame(
                batch_results_t.cpu().numpy(), columns=search_result_header
            )
            # todo for solved, log avg solution length , num_expanded, and num_generated
            batch_df["Time"] = batch_df["Time"].astype(float, copy=False) / 1000
            batch_df.sort_values("NumExpanded", inplace=True)
            solved_df = batch_df[batch_df["SolutionLength"] > 0]

            # avg_solution_length = solved_df["SolutionLength"].mean()
            # avg_num_expanded = solved_df["NumExpanded"].mean()
            # avg_num_generated = solved_df["NumGenerated"].mean()

            # if rank == 0:
            #     print(
            #         f"Avg solution length: {avg_solution_length}\nAvg num expanded: {avg_num_expanded}\nAvg num generated: {avg_num_generated}"
            #     )
            #     writer.add_scalar(
            #         "AvgSolutionLength", avg_solution_length, total_batches
            #     )
            #     writer.add_scalar("AvgNumExpanded", avg_num_expanded, total_batches)
            #     writer.add_scalar("AvgNumGenerated", avg_num_generated, total_batches)

            batch_solved_ids = solved_df["ProblemId"].values
            for problem_id in batch_solved_ids:
                if problem_id not in solved_problems:
                    num_new_problems_solved_this_epoch += 1
                    solved_problems.add(problem_id)

            num_problems_solved_this_batch = len(batch_solved_ids)
            num_problems_solved_this_epoch += num_problems_solved_this_batch
            if rank == 0:
                print(
                    tabulate(batch_df, headers="keys", tablefmt="psql", showindex=False)
                )
                print(f"Solved {num_problems_solved_this_batch}/{batch_size}\n")

            def fit_model(
                model: to.nn.Module,
                optimizer: to.optim.Optimizer,
                trajs: list,
                opt_step: int,
                name: str = "model",
            ):
                if rank == 0 and trajs:
                    print(opt_result_header)

                merged_traj = MergedTrajectory(trajs, shuffle_trajectory)
                to.set_grad_enabled(True)
                model.train()

                num_batch_partials = 0
                for _ in range(grad_steps):
                    optimizer.zero_grad()
                    if trajs:
                        loss, avg_action_nll, logits = loss_fn(merged_traj, model)
                        loss.backward()

                    if world_size > 1:
                        sync_grads(model, world_size)
                    # todo grad clipping?

                    optimizer.step()

                    if trajs:
                        avg_action_nll = avg_action_nll.item()
                        with to.no_grad():
                            acc = (
                                (logits.argmax(dim=1) == merged_traj.actions).sum()
                                / len(logits)
                            ).item()

                        local_opt_results[0] = avg_action_nll
                        local_opt_results[1] = acc
                        local_opt_results[2] = 1
                    else:
                        local_opt_results[0] = 0
                        local_opt_results[1] = 0
                        local_opt_results[2] = 0

                    if world_size > 1:
                        dist.all_reduce(local_opt_results, op=dist.ReduceOp.SUM)

                    num_batch_partials = local_opt_results[2].item()
                    if num_batch_partials > 0:
                        opt_step += 1
                        if rank == 0:
                            local_opt_step = opt_step % grad_steps
                            batch_opt_step = opt_step // grad_steps
                            if local_opt_step == 1 or local_opt_step == 0:
                                loss = local_opt_results[0].item() / num_batch_partials
                                acc = local_opt_results[1].item() / num_batch_partials
                                print(f"{opt_step:7}  {loss:5.3f}  {acc:5.3f}")
                                if local_opt_step == 0:
                                    # fmt: off
                                    writer.add_scalar( f"loss_vs_opt_step/{name}", loss, batch_opt_step,)
                                    writer.add_scalar( f"acc_vs_opt_step/{name}", acc, batch_opt_step,)
                                    # fmt:on

                if rank == 0 and num_batch_partials > 0:
                    if track_params and total_batches % 10 == 0:
                        for (
                            param_name,
                            param,
                        ) in forward_model.named_parameters():
                            writer.add_histogram(
                                f"param_vs_opt_step/{name}/{param_name}",
                                param.data,
                                batch_opt_step,
                                bins=512,
                            )
                    print("")
                return opt_step

            forward_opt_steps = fit_model(
                forward_model,
                forward_optimizer,
                forward_trajs,
                forward_opt_steps,
                name="forward",
            )
            if bidirectional:
                backward_opt_steps = fit_model(
                    backward_model,  # type:ignore
                    backward_optimizer,  # type:ignore
                    backward_trajs,  # type:ignore
                    backward_opt_steps,  # type:ignore
                    name="backward",
                )

            if rank == 0:
                to.save(forward_model, forward_model_path)  # type:ignore
                if bidirectional:
                    to.save(backward_model, backward_model_path)  # type:ignore
                batch_avg = num_problems_solved_this_batch / batch_size
                # fmt: off
                writer.add_scalar(f"budget_{current_budget}/solved_vs_batch", batch_avg, total_batches)
                writer.add_scalar(f"cum_unique_solved_vs_batch", len(solved_problems), total_batches)
                # fmt: on

        if rank == 0:
            print(
                "============================================================================"
            )
            print(
                f"Completed epoch {epoch}, solved {num_problems_solved_this_epoch}/{num_problems} problems with budget {current_budget}\n"
                f"Solved {num_new_problems_solved_this_epoch} new problems, {num_problems - len(solved_problems)} remaining."
            )
            print(
                "============================================================================\n"
            )

            # fmt: off
            epoch_solved = num_problems_solved_this_epoch / num_problems
            writer.add_scalar("budget_vs_epoch", current_budget, epoch)
            writer.add_scalar(f"budget_{current_budget}/solved_vs_epoch", epoch_solved, epoch)
            writer.add_scalar("cum_unique_solved_vs_epoch", len(solved_problems), epoch)
            # fmt: on

        if num_new_problems_solved_this_epoch == 0:
            current_budget *= 2


class ProblemsBatchLoader:
    def __init__(
        self, problems: list[tuple[int, Domain]], batch_size: int, shuffle: bool = True
    ):
        self.problems = np.array(problems)
        self._len = len(problems)
        self._num_problems_served = 0
        self._batch_size = batch_size
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
            self._num_problems_served : self._num_problems_served + self._batch_size
        ]
        self._num_problems_served += self._batch_size
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
