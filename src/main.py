import datetime
import json
import os
from pathlib import Path
import pickle
import random
import time
from timeit import default_timer as timer

import numpy as np
from test import test
import torch as to
import torch.distributed as dist
import torch.multiprocessing as mp

from args import parse_args
from loaders import ProblemLoader
from search.bilevin import BiLevin
from search.levin import Levin
from search.utils import set_seeds
from train import train


def split_by_rank(args, problems):
    "split a list of lists of problems into a list of lists of problems per rank"
    rng = np.random.default_rng(args.seed)

    def split_by_rank(problems):
        ranks_x_problems = [[] for _ in range(args.world_size)]
        rank = 0
        for problem in problems:
            # this should be a redundant check, but just in case
            if problem.domain.is_goal(problem.domain.initial_state):
                raise Exception(f"Problem '{problem.id}' initial state is a goal state")

            ranks_x_problems[rank].append(problem)
            rank = (rank + 1) % args.world_size

        # ensure all ranks have same number of problems per stage
        n_largest_pset = len(ranks_x_problems[0])
        for pset in ranks_x_problems:
            if len(pset) < n_largest_pset:
                pset.append(rng.choice(problems))
            assert len(pset) == n_largest_pset

        return ranks_x_problems

    stages_x_problems = problems
    num_stages = len(stages_x_problems)

    # turn stages x problems into stages x ranks x problems
    stages_x_ranks_x_problems = []
    for stage in range(num_stages):
        stages_x_ranks_x_problems.append(split_by_rank(stages_x_problems[stage]))

    world_num_problems = 0
    ranks_x_stages_x_problems = []
    for rank in range(args.world_size):
        curr_stages_x_problems = []
        for stage in range(num_stages):
            probs = stages_x_ranks_x_problems[stage][rank]
            curr_stages_x_problems.append(probs)
            world_num_problems += len(probs)
        ranks_x_stages_x_problems.append(curr_stages_x_problems)

    return ranks_x_stages_x_problems, world_num_problems


def run(
    rank,
    run_name,
    agent,
    args,
    local_train_problems,
    world_num_train_problems,
    local_valid_problems,
    world_num_valid_problems,
):
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    dist.init_process_group(
        timeout=datetime.timedelta(seconds=86400),
        backend="gloo",
        rank=rank,
        world_size=args.world_size,
    )

    if args.mode == "test":
        run_name = f"test_{run_name}"

    local_seed = args.seed + rank
    set_seeds(local_seed)
    to.set_num_threads(1)

    train_loader = ProblemLoader(
        world_num_train_problems,
        local_train_problems,
        seed=local_seed,
        manual_advance=args.min_solve_ratio > 0
        and args.min_samples_per_stage is not None,
    )
    valid_loader = ProblemLoader(
        world_num_valid_problems, local_valid_problems, seed=local_seed
    )

    if args.n_solve_ratio < len(local_train_problems[0]):
        args.n_solve_ratio = len(local_train_problems[0])

    if args.mode == "train":
        train(
            args,
            rank,
            agent,
            train_loader,
            valid_loader,
            seed=local_seed,
        )

    elif args.mode == "test":
        test(
            args,
            rank,
            agent,
            local_train_problems,
            print_results=True,
            epoch=None,
        )

    if rank == 0:
        total_time = timer() - rel_start_time
        print(f"Total time: {total_time:.2f} seconds")
        with (logdir / "total_time_seconds.txt").open("w") as f:
            f.write(f"{total_time:.2f}")


if __name__ == "__main__":
    args = parse_args()
    if args.min_solve_ratio > 0 and args.min_samples_per_stage is None:
        raise ValueError(
            "Must provide --min-samples-per-stage when using --min-stage-solve-ratio"
        )

    pset_dict = pickle.load(args.problems_path.open("rb"))
    problems, world_num_problems = split_by_rank(args, pset_dict["problems"])

    abs_start_time = time.time()
    rel_start_time = timer()
    print(time.ctime(abs_start_time))
    set_seeds(args.seed)
    exp_name = f"_{args.exp_name}" if args.exp_name else ""

    problemset_params = f"{args.problems_path.parent.stem}-{args.problems_path.stem}"
    run_name = f"{pset_dict['domain_name']}-{problemset_params}_{args.agent}-e{args.expansion_budget}-t{args.time_budget}{exp_name}_{args.seed}_{int(abs_start_time)}"

    logdir = args.runsdir_path / run_name
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to {str(logdir)}")

    model_args = {
        "kernel_size": (2, 2),
        "num_filters": 32,
        "share_feature_net": args.share_feature_net,
        "forward_hidden_layers": args.forward_hidden_layers,
        "backward_hidden_layers": args.backward_hidden_layers,
        "state_t_width": pset_dict["state_t_width"],
        "state_t_depth": pset_dict["state_t_depth"],
        "num_actions": pset_dict["num_actions"],
        "in_channels": pset_dict["in_channels"],
        "kernel_depth": pset_dict["kernel_depth"],
        "requires_backward_goal": pset_dict["requires_backward_goal"],
    }

    if args.agent == "Levin":
        agent = Levin(logdir, args, model_args)
    elif args.agent == "BiLevin":
        agent = BiLevin(logdir, args, model_args)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    if args.valid_path:
        vset_dict = pickle.load(args.valid_path.open("rb"))
        valid_problems, world_num_valid_problems = split_by_rank(
            args, vset_dict["problems"]
        )
    else:
        valid_problems = None
        world_num_valid_problems = 0

    arg_dict = {
        k: (v if not isinstance(v, Path) else str(v)) for k, v in vars(args).items()
    }
    with (logdir / "args.json").open("w") as f:
        json.dump(arg_dict, f, indent=2)
    for k, v in arg_dict.items():
        print(f"{k}: {v}")

    args.logdir = logdir

    processes = []
    for rank in range(args.world_size):
        proc = mp.Process(
            target=run,
            args=(
                rank,
                run_name,
                agent,
                args,
                problems[rank],
                world_num_problems,
                valid_problems[rank] if valid_problems else None,
                world_num_valid_problems if valid_problems else 0,
            ),
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()
