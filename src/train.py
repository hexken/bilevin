import os
import numpy as np
from pathlib import Path
import pathlib
import tqdm
import time

import torch as to
from torch.utils.data import Dataset, DataLoader

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
    initial_budget=7000,
    grad_steps=10,
    problems_batch_size=32,
    writer=None,
):

    forward_memory = Memory()
    bidirectional = isinstance(planner, tuple)
    if bidirectional:
        backward_memory = Memory()
        forward_model, backward_model = model
        forward_optimizer = optimizer_cons(
            forward_model.parameters(), **optimizer_params
        )
        backward_optimizer = optimizer_cons(
            backward_model.parameters(), **optimizer_params
        )
        backward_model_path = model_path[: len("_forward.pt")] + "_backward.pt"
    else:
        forward_model = model
        forward_optimizer = optimizer_cons(model.parameters(), **optimizer_params)

    current_solved_problems = set()
    num_expanded = 0
    num_generated = 0
    current_budget = initial_budget

    num_passes_problemset = 0
    opt_steps = 0
    num_problems = len(problems)
    num_problems_unsolved_total = num_problems

    problems_loader = ProblemsBatchLoader(
        problems, batch_size=problems_batch_size, shuffle=True
    )

    total_batches_seen = 0
    while num_problems_unsolved_total > 0:
        num_problems_solved_this_pass = 0
        for batch_problems, batch_names in tqdm.tqdm(problems_loader):
            num_problems_solved_this_batch = 0

            to.set_grad_enabled(False)
            forward_model.eval()
            if bidirectional:
                backward_model.eval()
            for problem, problem_name in zip(batch_problems, batch_names):
                start_time = time.time()
                (
                    has_found_solution,
                    num_expanded,
                    num_generated,
                    traj,
                ) = planner.search(
                    problem,
                    problem_name,
                    model,
                    current_budget,
                    learn=True,
                )

                if has_found_solution:
                    print(
                        f"{problem_name} SOLVED\n"
                        f"Time: {time.time() - start_time:.3f}\n"
                        f"Cost: {has_found_solution}\n"
                        f"Expanded: {num_expanded}\n"
                        f"Generated: {num_generated}\n"
                    )

                    forward_memory.add_trajectory(traj[0])
                    if bidirectional:
                        backward_memory.add_trajectory(traj[1])

                    if problem_name not in current_solved_problems:
                        num_problems_solved_this_pass += 1
                        num_problems_solved_this_batch += 1
                        current_solved_problems.add(problem_name)

            if forward_memory.number_trajectories() > 0:
                to.set_grad_enabled(True)
                forward_model.train()
                if bidirectional:
                    backward_model.train()

                for _ in range(grad_steps):
                    total_loss = 0
                    forward_memory.shuffle_trajectories()
                    for traj in forward_memory.next_trajectory():
                        forward_optimizer.zero_grad()
                        loss = loss_fn(traj, forward_model)
                        loss.backward()
                        forward_optimizer.step()
                        total_loss += loss.item()

                    avg_loss = total_loss / len(memory)
                    print(f"Avg Loss: {avg_loss:.3f}")
                    opt_steps += 1
                    writer.add_scalar(
                        "Loss/Train Avg (over memory)", avg_loss, opt_steps
                    )
                print("")
                forward_memory.clear()
                # todo log the losses and acc

                if bidirectional:
                    for _ in range(grad_steps):
                        total_loss = 0
                        backward_memory.shuffle_trajectories()
                        for traj in backward_memory.next_trajectory():
                            backward_optimizer.zero_grad()
                            loss = loss_fn(traj, backward_model)
                            loss.backward()
                            backward_optimizer.step()
                            total_loss += loss.item()

                        avg_loss = total_loss / len(backward_memory)
                        print(f"Avg Loss (B): {avg_loss:.3f}")
                        opt_steps += 1
                        writer.add_scalar("Loss/Train Avg (B)", avg_loss, opt_steps)
                    print("")
                    backward_memory.clear()

                # todo add save interval, performance increase check? validation? Overhaul checkpointing.
                if bidirectional:
                    to.save(forward_model, model_path)
                    to.save(backward_model, backward_model_path)
                else:
                    to.save(model, model_path)

            total_batches_seen += 1
            avg_problems_solved = num_problems_solved_this_batch / problems_batch_size
            # fmt: off
            writer.add_scalar("Search/Avg # Problems Solved vs. Batch", avg_problems_solved, total_batches_seen)
            writer.add_scalar("Search/# Problems Solved vs. Batch", num_problems_solved_this_pass, total_batches_seen)
            writer.add_scalar("Search/# Unsolved Provlems vs Batch", num_problems_unsolved_total, total_batches_seen)
            # fmt: on

        num_problems_unsolved_total -= num_problems_solved_this_pass
        num_passes_problemset += 1

        print("=========================================")
        print(
            f"Solved {num_problems_solved_this_pass} problem in pass #{num_passes_problemset} with budget {current_budget}"
        )
        print("=========================================\n")

        if num_problems_solved_this_pass == 0:
            current_budget *= 2
            print(f"Budget: {current_budget}")

        # fmt: off
        writer.add_scalar("Search/Budget vs. Pass", current_budget, num_passes_problemset)
        writer.add_scalar("Search/# Problems Solved vs. Pass", num_problems_solved_this_pass, num_passes_problemset)
        writer.add_scalar("Search/# Unsolved Problems vs. Pass", num_problems_unsolved_total, num_passes_problemset)
        # fmt: on
