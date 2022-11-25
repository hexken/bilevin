import os
from pathlib import Path
import pathlib
import tqdm
import time

import torch as to

from models.memory import Memory


class Bootstrap:
    def __init__(
        self,
        states,
        model_path,
        loss_fn,
        optimizer_cons,
        optimizer_params,
        initial_budget=2000,
        grad_steps=10,
        batch_size=32,
        writer=None,
    ):
        self._problems = states
        self._model_name = model_path
        self._loss_fn = loss_fn
        self._num_problems = len(states)
        self._optimizer_cons = optimizer_cons
        self._optimizer_params = optimizer_params

        self._initial_budget = initial_budget
        self._grad_steps = grad_steps
        self._batch_problems_size = batch_size

        self.model_path = model_path
        self.writer = writer

    def solve_uniform_online(self, planner, model):
        memory = Memory()
        optimizer = self._optimizer_cons(model.parameters(), **self._optimizer_params)

        current_solved_problems = set()
        num_expanded = 0
        num_generated = 0
        current_budget = self._initial_budget

        num_passes_problemset = 0
        opt_steps = 0
        num_problems_unsolved = self._num_problems

        while num_problems_unsolved > 0:
            num_problems_solved_this_pass = 0

            batch_problems = {}
            for problem_name, problem in tqdm.tqdm(self._problems.items()):

                batch_problems[problem_name] = problem

                if (
                    len(batch_problems)
                    < self._batch_problems_size
                    # and num_problems_unsolved > self._batch_problems_size
                ):
                    continue

                with to.no_grad():
                    model.eval()
                    for problem_name, problem in batch_problems.items():
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

                            memory.add_trajectory(traj)

                            if problem_name not in current_solved_problems:
                                num_problems_solved_this_pass += 1
                                current_solved_problems.add(problem_name)

                if memory.number_trajectories() > 0:
                    model.train()
                    # todo should we just do one opt step per sweep of memory (avg over all memory)?
                    for _ in range(self._grad_steps):
                        total_loss = 0
                        memory.shuffle_trajectories()
                        for traj in memory.next_trajectory():
                            optimizer.zero_grad()
                            loss = self._loss_fn(traj, model)
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()

                        avg_loss = total_loss / len(memory)
                        print(f"Avg Loss: {avg_loss:.3f}")
                        opt_steps += 1
                        self.writer.add_scalar(
                            "Loss/Train Avg (over memory)", avg_loss, opt_steps
                        )
                    print("")

                    memory.clear()
                    # todo add save interval, performance increase check? Overhaul checkpointing.
                    model.save_weights(self.model_path)

                batch_problems.clear()

            self.writer.add_scalar(
                "Performance/Num Problems Solved vs Problemset Passes",
                self._num_problems - num_problems_unsolved,
                num_passes_problemset,
            )
            self.writer.add_scalar(
                "Performance/Num Problems Solved vs Opt Steps",
                self._num_problems - num_problems_unsolved,
                opt_steps,
            )
            num_problems_unsolved -= num_problems_solved_this_pass
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
