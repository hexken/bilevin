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
        ncpus=1,
        initial_budget=2000,
        gradient_steps=10,
    ):
        self._states = states
        self._model_name = output
        self._loss_fn = loss_fn
        self._number_problems = len(states)
        self._optimizer_cons = optimizer_cons
        self._optimizer_params = optimizer_params

        self._ncpus = ncpus
        self._initial_budget = initial_budget
        self._gradient_steps = gradient_steps
        #         self._k = ncpus * 3
        self._batch_size = 32

        self._kmax = 10

        self._log_folder = "training_logs/"
        self._models_folder = "trained_models_online/" + self._model_name

        if not os.path.exists(self._models_folder):
            os.makedirs(self._models_folder)

        if not os.path.exists(self._log_folder):
            os.makedirs(self._log_folder)

    def solve_uniform_online(self, planner, model):
        memory = Memory()
        optimizer = self._optimizer_cons(model.parameters(), **self._optimizer_params)
        start = time.time()

        current_solved_problems = set()
        number_solved = 0
        total_expanded = 0
        total_generated = 0
        budget = self._initial_budget

        iteration = 1
        while len(current_solved_problems) < self._number_problems:
            number_solved = 0

            batch_problems = {}
            for problem_name, initial_state in self._states.items():

                #                 if name in current_solved_problems:
                #                     continue

                batch_problems[problem_name] = initial_state

                if (
                    len(batch_problems) < self._batch_size
                    and self._number_problems - len(current_solved_problems)
                    > self._batch_size
                ):
                    continue

                with to.no_grad():
                    model.eval()
                    for problem_name, initial_state in batch_problems.items():
                        (
                            has_found_solution,
                            trajectory,
                            total_expanded,
                            total_generated,
                        ) = planner.search_for_learning(
                            initial_state, problem_name, budget, model
                        )

                        if has_found_solution:
                            memory.add_trajectory(trajectory)

                            if problem_name not in current_solved_problems:
                                number_solved += 1
                                current_solved_problems.add(problem_name)

                if memory.number_trajectories() > 0:
                    model.train()
                    for _ in range(self._gradient_steps):
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
                    # todo add save interval
                    model.save_weights(join(self._models_folder, "model_weights"))

                batch_problems.clear()

            end = time.time()
            with open(
                join(self._log_folder + "training_bootstrap_" + self._model_name), "a"
            ) as results_file:
                results_file.write(
                    (
                        "{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:f} \n".format(
                            iteration,
                            number_solved,
                            self._number_problems - len(current_solved_problems),
                            budget,
                            total_expanded,
                            total_generated,
                            end - start,
                        )
                    )
                )

            print("Number solved: ", number_solved)
            if number_solved == 0:
                budget *= 2
                print("Budget: ", budget)
                continue

            iteration += 1
