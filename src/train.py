import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch as to
import torch.distributed as dist
from search import MergedTrajectories
import tqdm


def train(
    agent,
    model,
    model_path,
    loss_fn,
    optimizer_cons,
    optimizer_params,
    problems,
    writer,
    world_size=1,
    initial_budget=7000,
    grad_steps=10,
    batch_size=32,
):
    current_budget = initial_budget

    if world_size > 1:
        rank = dist.get_rank()
    else:
        rank = 0

    num_problems = len(problems) * world_size

    search_result_header = [
        "Problem",
        "SolutionLength",
        "NumExpanded",
        "NumGenerated",
        "Time",
    ]
    opt_result_header = f"OptStep   Loss    Acc"

    bidirectional = agent.bidirectional
    if bidirectional:
        forward_model, backward_model = model
        forward_optimizer = optimizer_cons(
            forward_model.parameters(), **optimizer_params
        )
        backward_optimizer = optimizer_cons(
            backward_model.parameters(), **optimizer_params
        )
        backward_model_path = Path(
            str(model_path).replace("_forward.pt", "_backward.pt")
        )
    else:
        forward_model = model
        forward_model_path = model_path
        forward_optimizer = optimizer_cons(model.parameters(), **optimizer_params)

    device = next(forward_model.parameters()).device

    local_batch_size = batch_size // world_size
    problems_loader = ProblemsBatchLoader(
        problems, batch_size=local_batch_size, shuffle=True
    )

    local_opt_results = to.zeros(5, dtype=to.float32)
    local_batch_results = to.zeros(local_batch_size, 5, dtype=to.int32).to(device)
    batch_results = [
        to.zeros((local_batch_size, 5), dtype=to.int32).to(device)
        for _ in range(world_size)
    ]

    total_batches = 0
    epoch = 0

    solved_problems = set()

    forward_opt_steps = 0
    backward_opt_steps = 0
    num_new_problems_solved = 0
    while num_new_problems_solved < num_problems:
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
                problems_loader, total=math.ceil(len(problems_loader) / batch_size)
            )

        for batch_problems, batch_ids in problems_loader:
            total_batches += 1

            if rank == 0:
                print(f"\n\nBatch {total_batches}\n")

            forward_solutions = []
            to.set_grad_enabled(False)
            forward_model.eval()
            if bidirectional:
                backward_solutions = []
                backward_model.eval()  # type:ignore

            num_problems_solved_this_batch = 0
            for i, (problem, problem_id) in enumerate(zip(batch_problems, batch_ids)):
                start_time = time.time()
                (solution_length, num_expanded, num_generated, traj,) = agent.search(
                    problem,
                    problem_id,
                    model,
                    current_budget,
                    learn=True,
                )

                local_batch_results[i, 0] = int(problem_id[1:])
                local_batch_results[i, 1] = solution_length
                local_batch_results[i, 2] = num_expanded
                local_batch_results[i, 3] = num_generated
                local_batch_results[i, 4] = int((time.time() - start_time) * 1000)

                if solution_length:
                    forward_solutions.append(traj[0])
                    if bidirectional:
                        btraj = traj[1]
                        dims = [1] * btraj.states.dim()
                        goal_repeated = btraj.goal.repeat(len(btraj.states), *dims)
                        btraj_states_with_goal = to.vstack(
                            (btraj.states, goal_repeated)
                        )
                        btraj.states = btraj_states_with_goal
                        backward_solutions.append(btraj)

            if world_size > 1:
                dist.all_gather(batch_results, local_batch_results)
                batch_results_t = to.cat(batch_results, dim=0)
            else:
                batch_results_t = local_batch_results

            batch_df = pd.DataFrame(
                batch_results_t,
                columns=search_result_header,
            )
            batch_df["Time"] = batch_df["Time"].astype(float, copy=False) / 1000
            batch_df.sort_values("NumExpanded", inplace=True)

            batch_solved_ids = batch_df[batch_df.SolutionLength > 0].Problem.values
            for problem_id in batch_solved_ids:
                if problem_id not in solved_problems:
                    num_new_problems_solved_this_epoch += 1
                    solved_problems.add(problem_id)

            num_problems_solved_this_batch = len(batch_solved_ids)
            num_problems_solved_this_epoch += num_problems_solved_this_batch
            if rank == 0:
                print(batch_df)
                print(f"Solved {num_problems_solved_this_batch}/{batch_size}\n")

            def fit_model(model, optimizer, solutions, opt_step):
                if rank == 0 and solutions:
                    print(opt_result_header)

                merged_solutions = MergedTrajectories(solutions)
                to.set_grad_enabled(True)
                model.train()

                for _ in range(grad_steps):
                    optimizer.zero_grad()
                    if solutions:
                        loss, logits = loss_fn(merged_solutions, model)
                        loss.backward()

                    if world_size > 1:
                        sync_grads(model, world_size)
                    # todo grad clipping?

                    optimizer.step()

                    if solutions:
                        loss = loss.item()
                        with to.no_grad():
                            acc = (
                                (logits.argmax(dim=1) == merged_solutions.actions).sum()
                                / len(logits)
                            ).item()

                        local_opt_results[0] = loss
                        local_opt_results[1] = acc
                        local_opt_results[2] = 1
                    else:
                        local_opt_results[0] = 0
                        local_opt_results[1] = 0
                        local_opt_results[2] = 0

                    if world_size > 1:
                        dist.all_reduce(local_opt_results, op=dist.ReduceOp.SUM)

                    # todo double check
                    num_batch_partials = local_opt_results[2].item()
                    if rank == 0 and num_batch_partials > 0:
                        opt_step += 1
                        loss = local_opt_results[0].item() / num_batch_partials
                        acc = local_opt_results[1].item() / num_batch_partials
                        print(f"{opt_step:7}  {loss:5.3f}  {acc:5.3f}")
                        # fmt: off
                        writer.add_scalar( "loss/forward/loss_vs_opt_step", loss, opt_step,)
                        writer.add_scalar( "accuracy/forward/acc_vs_opt_step", acc, opt_step,)
                        # fmt:on
                if rank == 0:
                    print("")
                return opt_step

            forward_opt_steps = fit_model(
                forward_model, forward_optimizer, forward_solutions, forward_opt_steps
            )
            if bidirectional:
                backward_opt_steps = fit_model(
                    backward_model,  # type:ignore
                    backward_optimizer,  # type:ignore
                    backward_solutions,  # type:ignore
                    backward_opt_steps,  # type:ignore
                )

            if rank == 0:
                to.save(forward_model, forward_model_path)  # type:ignore
                if bidirectional:
                    to.save(forward_model, backward_model_path)  # type:ignore

            if rank == 0:
                batch_avg = num_problems_solved_this_batch / batch_size
                # fmt: off
                writer.add_scalar(f"search/budget_{current_budget}/batch_solved_vs_batch", batch_avg, total_batches)
                # writer.add_scalar("Search/cumulative_unique_problems_solved_vs_batch", len(solved_problems), total_batches)
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
            writer.add_scalar("search/budget_vs_epoch", current_budget, epoch)
            writer.add_scalar(f"search/budget_{current_budget}/solved_vs_epoch", epoch_solved, epoch)
            writer.add_scalar("search/cumulative_unique_problems_solved_vs_epoch", len(solved_problems), epoch)
            # fmt: on

        if num_new_problems_solved_this_epoch == 0:
            current_budget *= 2


class ProblemsBatchLoader:
    def __init__(self, problems, batch_size, shuffle=True):
        self.problems = np.array(tuple(problems.values()))
        self.keys = np.array(tuple(problems.keys()))
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
        return self.problems[next_indices], self.keys[next_indices]

    def __getitem__(self, idx):
        return self.problems[idx], self.keys[idx]


def sync_grads(model: to.nn.Module, world_size):
    # todo some issues here
    all_grads_list = []
    for param in model.parameters():
        if param.grad is not None:
            all_grads_list.append(param.grad.view(-1))
        else:
            zeros = param.data.new(param.data.shape).zero_()
            all_grads_list.append(zeros.view(-1))
    all_grads = to.cat(all_grads_list)
    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
    offset = 0
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.copy_(
                all_grads[offset : offset + param.numel()].view_as(param.grad.data)
                / world_size
            )
        else:
            param.grad = (
                all_grads[offset : offset + param.numel()].view_as(param.data)
                / world_size
            ).copy()
            offset += param.numel()
