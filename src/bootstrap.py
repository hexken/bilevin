import os
from os.path import join
import time

import torch as to

from models.memory import Memory


class Bootstrap:
    def __init__(
        self,
        states,
        output,
        loss_fn,
        optimizer_cons,
        optimizer_params,
        initial_budget=2000,
        grad_steps=10,
        writer=None,
    ):
        self._states = states
        self._model_name = output
        self._loss_fn = loss_fn
        self._num_problems = len(states)
        self._optimizer_cons = optimizer_cons
        self._optimizer_params = optimizer_params

        self._initial_budget = initial_budget
        self._grad_steps = grad_steps
        self._batch_size_expansions = 32

        self._log_folder = "training_logs/"
        self._models_folder = "trained_models_online/" + self._model_name

        if not os.path.exists(self._models_folder):
            os.makedirs(self._models_folder)

        if not os.path.exists(self._log_folder):
            os.makedirs(self._log_folder)

    def solve_uniform_online(self, planner, model):
        memory = Memory()
        optimizer = self._optimizer_cons(model.parameters(), **self._optimizer_params)
        start_time = time.time()

        current_solved_problems = set()
        num_expanded = 0
        num_generated = 0
        current_budget = self._initial_budget

        iteration = 1
        num_unsolved_problems = self._num_problems
        while num_unsolved_problems > 0:
            num_problems_solved_this_iter = 0

            batch_problems = {}
            for problem_name, initial_state in self._states.items():

                batch_problems[problem_name] = initial_state

                if (
                    len(batch_problems) < self._batch_size_expansions
                    and self._num_problems - len(current_solved_problems)
                    > self._batch_size_expansions
                ):
                    continue

                with to.no_grad():
                    model.eval()
                    for problem_name, initial_state in batch_problems.items():
                        (
                            has_found_solution,
                            num_expanded,
                            num_generated,
                            traj,
                        ) = planner.search(
                            initial_state,
                            problem_name,
                            model,
                            current_budget,
                            learn=True,
                        )

                        if has_found_solution:
                            memory.add_trajectory(traj)

                            if problem_name not in current_solved_problems:
                                num_problems_solved_this_iter += 1
                                current_solved_problems.add(problem_name)

                if memory.number_trajectories() > 0:
                    model.train()
                    for _ in range(self._grad_steps):
                        total_loss = 0
                        memory.shuffle_trajectories()
                        for traj in memory.next_trajectory():
                            optimizer.zero_grad()
                            loss = self._loss_fn(traj, model)
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()

                        print("Avg Loss: ", total_loss / len(memory))
                    memory.clear()
                    # todo add save interval, overhaul checkpointing
                    model.save_weights(join(self._models_folder, "model_weights.pt"))

                batch_problems.clear()

            with open(
                join(self._log_folder + "training_bootstrap_" + self._model_name), "a"
            ) as results_file:
                results_file.write(
                    (
                        "{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:f} \n".format(
                            iteration,
                            num_problems_solved_this_iter,
                            self._num_problems - len(current_solved_problems),
                            current_budget,
                            num_expanded,
                            num_generated,
                            time.time() - start_time,
                        )
                    )
                )
            num_unsolved_problems -= num_problems_solved_this_iter

            print(f"Number solved: {num_problems_solved_this_iter}")
            if num_problems_solved_this_iter == 0:
                current_budget *= 2
                print(f"Budget: {current_budget}")
                continue

            iteration += 1
