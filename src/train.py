import os
from pathlib import Path
import time

import numpy as np
import torch as to
from torch.utils.data import DataLoader, Dataset
import tqdm

from search.utils import Memory


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


def train(
    problems,
    model,
    model_path,
    planner,
    loss_fn,
    optimizer_cons,
    optimizer_params,
    writer,
    initial_budget=7000,
    grad_steps=10,
    problems_batch_size=32,
):

    forward_memory = Memory()
    bidirectional = planner.bidirectional
    if bidirectional:
        backward_memory = Memory()
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
        forward_optimizer = optimizer_cons(model.parameters(), **optimizer_params)

    solved_problems = set()
    num_expanded = 0
    num_generated = 0
    current_budget = initial_budget

    problems_loader = ProblemsBatchLoader(
        problems, batch_size=problems_batch_size, shuffle=True
    )

    forward_outer_opt_steps = 0
    # forward_inner_opt_steps = 0
    backward_outer_opt_steps = 0
    # backward_inner_opt_steps = 0
    num_problems = len(problems)
    total_batches = 0
    epoch = 0
    cumulative_unique_problems_solved = 0

    while cumulative_unique_problems_solved < num_problems:
        epoch += 1
        num_new_problems_solved_this_epoch = 0
        num_problems_solved_this_epoch = 0

        for batch_problems, batch_names in tqdm.tqdm(problems_loader):
            total_batches += 1
            num_problems_solved_this_batch = 0

            to.set_grad_enabled(False)
            forward_model.eval()
            if bidirectional:
                backward_model.eval()  # type:ignore

            for problem, problem_name in zip(batch_problems, batch_names):
                start_time = time.time()
                (
                    has_found_solution,
                    num_expanded,
                    num_generated,
                    trajs,
                ) = planner.search(
                    problem,
                    problem_name,
                    model,
                    current_budget,
                    learn=True,
                )

                if has_found_solution:
                    num_problems_solved_this_batch += 1
                    print(
                        f"{problem_name} SOLVED\n"
                        f"Time: {time.time() - start_time:.3f}\n"
                        f"Cost: {has_found_solution}\n"
                        f"Expanded: {num_expanded}\n"
                        f"Generated: {num_generated}\n"
                    )

                    forward_memory.add_trajectory(trajs[0])
                    if bidirectional:
                        btraj = trajs[1]
                        btraj_states_with_goal = to.vstack(
                            (btraj.states, btraj.goal.repeat(len(btraj)))
                        )
                        btraj.states = btraj_states_with_goal
                        backward_memory.add_trajectory(btraj)  # type:ignore

                    if problem_name not in solved_problems:
                        num_new_problems_solved_this_epoch += 1
                        solved_problems.add(problem_name)

            num_problems_solved_this_epoch += num_problems_solved_this_batch
            if forward_memory.number_trajectories() > 0:
                to.set_grad_enabled(True)
                forward_model.train()
                if bidirectional:
                    backward_model.train()  # type:ignore

                for _ in range(grad_steps):
                    total_loss = 0
                    forward_memory.shuffle_trajectories()
                    for trajs in forward_memory.next_trajectory():
                        forward_optimizer.zero_grad()
                        loss = loss_fn(trajs, forward_model)
                        loss.backward()
                        forward_optimizer.step()
                        total_loss += loss.item()

                    avg_loss = total_loss / len(forward_memory)
                    print(f"Avg Loss (F): {avg_loss:.3f}")
                    forward_outer_opt_steps += 1
                    writer.add_scalar(
                        "Loss/f_train_avg_traj_loss_over_memory_vs_outeropt",
                        avg_loss,
                        forward_outer_opt_steps,
                    )
                print("")
                forward_memory.clear()

                if bidirectional:
                    for _ in range(grad_steps):
                        total_loss = 0
                        backward_memory.shuffle_trajectories()  # type:ignore
                        for trajs in backward_memory.next_trajectory():  # type:ignore
                            backward_optimizer.zero_grad()  # type:ignore
                            loss = loss_fn(trajs, backward_model)  # type:ignore
                            loss.backward()
                            backward_optimizer.step()  # type:ignore
                            total_loss += loss.item()

                        avg_loss = total_loss / len(backward_memory)  # type:ignore
                        print(f"Avg Loss (B): {avg_loss:.3f}")
                        backward_outer_opt_steps += 1
                        writer.add_scalar(
                            "Loss/f_train_avg_traj_loss_over_memory_vs_outeropt",
                            avg_loss,
                            backward_outer_opt_steps,
                        )
                    print("")
                    backward_memory.clear()  # type:ignore

                # todo add save interval, performance increase check? validation? Overhaul checkpointing.
                if bidirectional:
                    to.save(forward_model, model_path)
                    to.save(backward_model, backward_model_path)  # type:ignore
                else:
                    to.save(model, model_path)

            batch_avg = num_problems_solved_this_batch / problems_batch_size
            # fmt: off
            writer.add_scalar("Search/avg_batch_solved_vs_batch", batch_avg, total_batches)
            writer.add_scalar("Search/cumulative_unique_problems_solved_vs_batch", cumulative_unique_problems_solved, total_batches)
            # fmt: on

        cumulative_unique_problems_solved += num_new_problems_solved_this_epoch

        print("=========================================")
        print(
            f"Solved {num_problems_solved_this_epoch}/{num_problems} problems in pass #{epoch} with budget {current_budget}"
        )
        print("=========================================\n")

        if num_new_problems_solved_this_epoch == 0:
            current_budget *= 2
            print(f"Budget: {current_budget}")

        # fmt: off
        epoch_avg = num_problems_solved_this_epoch / num_problems
        writer.add_scalar("Search/budget_vs_epoch", current_budget, epoch)
        writer.add_scalar("Search/avg_solved_vs_epoch", epoch_avg, epoch)
        writer.add_scalar("Search/cumulative_unique_problems_solved_vs_epoch", cumulative_unique_problems_solved, epoch)
        # fmt: on
