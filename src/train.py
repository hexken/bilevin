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

    bidirectional = planner.bidirectional

    memory = Memory()
    if bidirectional:
        backward_memory = Memory()
        forward_model, backward_model = model
        optimizer = optimizer_cons(forward_model.parameters(), **optimizer_params)
        backward_optimizer = optimizer_cons(
            backward_model.parameters(), **optimizer_params
        )
    else:
        optimizer = optimizer_cons(model.parameters(), **optimizer_params)

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

    while num_problems_unsolved_total > 0:
        num_problems_solved_this_pass = 0
        for batch_problems, batch_names in tqdm.tqdm(problems_loader):

            to.set_grad_enabled(False)
            if bidirectional:
                forward_model.eval()
                backward_model.eval()
            else:
                model.eval()
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

                    if bidirectional:
                        memory.add_trajectory(traj[0])
                        backward_memory.add_trajectory(traj[1])
                    else:
                        memory.add_trajectory(traj)

                    if problem_name not in current_solved_problems:
                        num_problems_solved_this_pass += 1
                        current_solved_problems.add(problem_name)

            if memory.number_trajectories() > 0:
                to.set_grad_enabled(True)
                model.train()
                if bidirectional:
                    forward_model.eval()
                    backward_model.eval()
                else:
                    model.eval()

                for _ in range(grad_steps):
                    total_loss = 0
                    memory.shuffle_trajectories()
                    for traj in memory.next_trajectory():
                        optimizer.zero_grad()
                        loss = loss_fn(traj, model.forward_model)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()

                    avg_loss = total_loss / len(memory)
                    print(f"Avg Loss: {avg_loss:.3f}")
                    opt_steps += 1
                    writer.add_scalar(
                        "Loss/Train Avg (over memory)", avg_loss, opt_steps
                    )
                print("")
                memory.clear()

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

                # todo add save interval, performance increase check? Overhaul checkpointing.
                model.save_weights(model_path)

        writer.add_scalar(
            "Performance/Num Problems Solved vs Problemset Passes",
            num_problems - num_problems_unsolved_total,
            num_passes_problemset,
        )
        writer.add_scalar(
            "Performance/Num Problems Solved vs Opt Steps",
            num_problems - num_problems_unsolved_total,
            opt_steps,
        )
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
            continue
