import datetime
import json
import os
from pathlib import Path
import pickle
import time
from timeit import default_timer as timer

import numpy as np
from test import test
import torch as to
import torch.distributed as dist
import torch.multiprocessing as mp

from args import parse_args
from loaders import ProblemLoader
from search.levin import Levin, BiLevin
from search.astar import AStar, BiAStar
from search.utils import set_seeds
from train import train


def split_by_rank(args, problems):
    "split a list of lists of problems into a list of lists of problems per rank"
    rng = np.random.default_rng(args.seed)

    def split_by_rank(problems):
        ranks_x_problems = [[] for _ in range(args.world_size)]
        rank = 0
        for problem in problems:
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
    agent,
    args,
    local_problems,
    world_num_problems,
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

    local_seed = args.seed + rank
    set_seeds(local_seed)
    to.set_num_threads(1)

    problems_loader = ProblemLoader(
        world_num_problems,
        local_problems,
        seed=local_seed,
    )

    if args.mode == "train":
        valid_loader = ProblemLoader(
            world_num_valid_problems,
            local_valid_problems,
            seed=local_seed,
            shuffle=False,
        )
        if rank == 0:
            print("\nTraining...")
        train(
            args,
            rank,
            agent,
            problems_loader,
            valid_loader,
        )
    else:
        if rank == 0:
            print("\nTesting...")
        test(
            args,
            rank,
            agent,
            problems_loader,
            print_results=True,
            batch=None,
        )

    if rank == 0:
        total_time = timer() - rel_start_time
        print(f"Finished!\nTotal time: {total_time:.2f} seconds")
        with (logdir / "total_time_seconds.txt").open("w") as f:
            f.write(f"{total_time:.2f}")


if __name__ == "__main__":
    args = parse_args()
    abs_start_time = time.time()
    rel_start_time = timer()
    print(time.ctime(abs_start_time))
    set_seeds(args.seed)
    exp_name = f"_{args.exp_name}" if args.exp_name else ""

    pset_dict = pickle.load(args.problems_path.open("rb"))
    problems, world_num_problems = split_by_rank(args, pset_dict["problems"])

    if args.mode == "train":
        if args.checkpoint_path is None:
            run_name = f"{args.problems_path.parent.stem}-{args.problems_path.stem}_{args.agent}_e{args.train_expansion_budget}_t{args.time_budget}{exp_name}_{args.seed}_{int(abs_start_time)}"
            logdir = args.runsdir_path / run_name
        else:
            run_name = str(args.checkpoint_path.parent)
            logdir = args.checkpoint_path.parent
            print(f"Loaded checkpoint {str(args.checkpoint_path)}")
    elif args.mode == "test":
        if args.model_path is not None:
            logdir = args.model_path.parent
            model_name = args.model_path.stem
        elif args.checkpoint_path is not None:
            logdir = args.checkpoint_path.parent
            model_name = args.checkpoint_path.stem
            print(f"Loaded checkpoint {str(args.checkpoint_path)}")
        else:
            raise ValueError(
                "Must specify either model_path or checkpoint_path to test"
            )
        logdir /= f"test_{model_name}_{args.seed}_{int(abs_start_time)}"
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    logdir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to {str(logdir)}")

    model_args = {
        "share_feature_net": args.share_feature_net,
        "no_feature_net": args.no_feature_net,
        "conditional_backward": args.conditional_backward,
        "forward_hidden_layers": args.forward_hidden_layers,
        "backward_hidden_layers": args.backward_hidden_layers,
        "state_t_width": pset_dict["state_t_width"],
        "state_t_depth": pset_dict["state_t_depth"],
        "num_actions": pset_dict["num_actions"],
        "in_channels": pset_dict["in_channels"],
        "kernel_size": (2, 2),
        "num_filters": 32,
        "kernel_depth": pset_dict["kernel_depth"],
    }

    if args.agent == "Levin":
        agent = Levin(logdir, args, model_args)
    elif args.agent == "BiLevin":
        agent = BiLevin(logdir, args, model_args)
    elif args.agent == "AStar":
        agent = BiLevin(logdir, args, model_args)
    elif args.agent == "BiAStar":
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
    argspath = logdir / "args.json"
    if not argspath.is_file():
        with argspath.open("w") as f:
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
