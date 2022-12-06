import math
from pathlib import Path
import time

import numpy as np
import torch as to
import torch.distributed as dist
import tqdm

from search import Memory


def train(
    rank,
    world_size,
    problems,
    model,
    model_path,
    agent,
    loss_fn,
    optimizer_cons,
    optimizer_params,
    writer,
    initial_budget=7000,
    grad_steps=10,
    batch_size=32,
):

    forward_memory = Memory()
    bidirectional = agent.bidirectional
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

    local_batch_size = batch_size // world_size
    problems_loader = ProblemsBatchLoader(
        problems, batch_size=local_batch_size, shuffle=True
    )

    forward_memory_passes = 0
    forward_opt_steps = 0
    backward_memory_passes = 0
    backward_opt_steps = 0
    num_problems = len(problems)
    total_batches = 0
    epoch = 0

    if world_size > 1:
        data_store = dist.TCPStore("127.0.0.1", 29500, world_size, rank == 0)

    while len(solved_problems) < num_problems:
        epoch += 1
        num_new_problems_solved_this_epoch = 0
        num_problems_solved_this_epoch = 0

        if rank == 0:
            problems_loader = tqdm.tqdm(
                problems_loader, total=math.ceil(len(problems_loader) / batch_size)
            )

        for batch_problems, batch_names in problems_loader:
            total_batches += 1
            num_problems_solved_this_batch = 0

            if rank == 0:
                print(f"\n\nBatch {total_batches}\n")

            to.set_grad_enabled(False)
            forward_model.eval()
            if bidirectional:
                backward_model.eval()  # type:ignore

            batch_results = to.zeros(batch_size, 5, dtype=to.long)
            for problem, problem_name in zip(batch_problems, batch_names):
                start_time = time.time()
                (solution_length, num_expanded, num_generated, traj,) = agent.search(
                    problem,
                    problem_name,
                    model,
                    current_budget,
                    learn=True,
                )

                if solution_length:
                    batch_results[num_problems_solved_this_batch][0] = int(
                        problem_name[1:]
                    )
                    batch_results[num_problems_solved_this_batch][1] = solution_length
                    batch_results[num_problems_solved_this_batch][2] = (
                        time.time() - start_time
                    )
                    batch_results[num_problems_solved_this_batch][3] = num_expanded
                    batch_results[num_problems_solved_this_batch][4] = num_generated
                    num_problems_solved_this_batch += 1

                    forward_memory.add_trajectory(traj[0])
                    if bidirectional:
                        btraj = traj[1]
                        dims = [1] * btraj.states.dim()
                        goal_repeated = btraj.goal.repeat(len(btraj.states), *dims)
                        btraj_states_with_goal = to.vstack(
                            (btraj.states, goal_repeated)
                        )
                        btraj.states = btraj_states_with_goal
                        backward_memory.add_trajectory(btraj)  # type:ignore

            if world_size > 1:
                if rank == 0:
                    problem_names = []
                    solution_lengths = []
                    times = []
                    nums_expanded = []
                    nums_generated = []

                    for i in range(batch_size):
                        if batch_results[i][1] == 0:
                            break
                        problem_names.append(batch_results[i][0])
                        solution_lengths.append(batch_results[i][1])
                        times.append(batch_results[i][2])
                        nums_expanded.append(batch_results[i][3])
                        nums_generated.append(batch_results[i][4])

                    for wait_rank in range(world_size):
                        dist.recv(tensor=batch_results, src=wait_rank)

                    if problem_name not in solved_problems:
                        num_new_problems_solved_this_epoch += 1
                        solved_problems.add(problem_name)

                else:
                    dist.send(tensor=batch_results, dst=0)
            else:
                pass

            num_problems_solved_this_epoch += num_problems_solved_this_batch
            print(f"Solved {num_problems_solved_this_batch}/{batch_size}\n")
            if forward_memory.number_trajectories() > 0:
                to.set_grad_enabled(True)
                forward_model.train()
                for _ in range(grad_steps):
                    total_loss = 0
                    forward_memory.shuffle_trajectories()
                    for traj in forward_memory.next_trajectory():
                        forward_optimizer.zero_grad()
                        loss, logits = loss_fn(traj, forward_model)
                        loss.backward()
                        forward_optimizer.step()
                        loss = loss.item()
                        total_loss += loss
                        writer.add_scalar(
                            "loss/forward/loss_vs_opt_step",
                            loss,
                            forward_opt_steps,
                        )
                        with to.no_grad():
                            num_solved = (
                                (logits.argmax(dim=1) == traj.actions).sum().item()
                            )

                    avg_loss = total_loss / len(forward_memory)
                    print(f"Loss (F) {avg_loss:.3f}")
                    forward_memory_passes += 1
                    writer.add_scalar(
                        "loss/forward/loss_vs_memory_pass",
                        avg_loss,
                        forward_memory_passes,
                    )
                print("")
                forward_memory.clear()

                if bidirectional:
                    backward_model.train()  # type:ignore
                    for _ in range(grad_steps):
                        total_loss = 0
                        backward_memory.shuffle_trajectories()  # type:ignore
                        for traj in backward_memory.next_trajectory():  # type:ignore
                            backward_optimizer.zero_grad()  # type:ignore
                            loss = loss_fn(traj, backward_model)  # type:ignore
                            loss.backward()
                            backward_optimizer.step()  # type:ignore
                            loss = loss.item()
                            total_loss += loss
                            writer.add_scalar(
                                "loss/forward/loss_vs_opt_step",
                                loss,
                                backward_opt_steps,
                            )

                        avg_loss = total_loss / len(backward_memory)  # type:ignore
                        print(f"Loss (B) {avg_loss:.3f}")
                        backward_memory_passes += 1
                        writer.add_scalar(
                            "loss/backward/loss_vs_memory_pass",
                            avg_loss,
                            backward_memory_passes,
                        )
                    print("")
                    backward_memory.clear()  # type:ignore

                if bidirectional:
                    to.save(forward_model, model_path)
                    to.save(backward_model, backward_model_path)  # type:ignore
                else:
                    to.save(model, model_path)

            batch_avg = num_problems_solved_this_batch / batch_size
            # fmt: off
            writer.add_scalar(f"search/budget_{current_budget}/batch_solved_vs_batch", batch_avg, total_batches)
            # writer.add_scalar("Search/cumulative_unique_problems_solved_vs_batch", len(solved_problems), total_batches)
            # fmt: on

        print(
            "============================================================================"
        )
        print(
            f"Epoch {epoch}, solved {num_problems_solved_this_epoch}/{num_problems} problems with budget {current_budget}\n"
            f"Solved {num_new_problems_solved_this_epoch} new problems, {num_problems - len(solved_problems)} remaining."
        )
        print(
            "============================================================================\n"
        )

        if num_new_problems_solved_this_epoch == 0:
            current_budget *= 2
            print(f"Budget: {current_budget}")

        # fmt: off
        epoch_solved = num_problems_solved_this_epoch / num_problems
        writer.add_scalar("search/budget_vs_epoch", current_budget, epoch)
        writer.add_scalar(f"search/budget_{current_budget}/solved_vs_epoch", epoch_solved, epoch)
        writer.add_scalar("search/cumulative_unique_problems_solved_vs_epoch", len(solved_problems), epoch)
        # fmt: on


class ResultData:
    def __init__(self, problem_name, cost, time, num_expanded, num_generated):
        self.problem_name = problem_name
        self.cost = cost
        self.time = time
        self.num_expanded = num_expanded
        self.num_generated = num_generated


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
