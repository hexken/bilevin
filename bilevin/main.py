from copy import deepcopy
import datetime
import json
import os
from pathlib import Path
import pickle as pkl
import time
from timeit import default_timer as timer

import numpy as np
import torch as to
import torch.distributed as dist
import torch.multiprocessing as mp

from args import parse_args
import search.agents as sa
from search.loaders import AsyncProblemLoader, ProblemLoader
from search.utils import set_seeds
from test_async import test
from train_async import train
from utils import find_free_port, split_by_rank


def run(
    rank,
    agent,
    args,
    problems,
    valid_problems,
    results_queue,
):
    dist.init_process_group(
        timeout=datetime.timedelta(seconds=86400),
        backend="gloo",
        rank=rank,
        world_size=args.world_size,
    )

    local_seed = args.seed + rank
    set_seeds(local_seed)
    to.set_num_threads(1)

    if args.mode == "train":
        if rank == 0:
            print(f"\nTraining on {len(problems)} problems for {args.n_epochs} epochs")
        train(
            args,
            rank,
            agent,
            problems,
            valid_problems,
            results_queue,
        )
        (logdir / "training_completed.txt").open("w").close()
    else:
        if rank == 0:
            print("\nTesting...")
        test(
            args,
            rank,
            agent,
            problems,
            results_queue,
            print_results=True,
        )


if __name__ == "__main__":
    args = parse_args()
    abs_start_time = time.time()
    rel_start_time = timer()
    print(time.ctime(abs_start_time))
    set_seeds(args.seed)
    exp_name = f"_{args.exp_name}" if args.exp_name else ""

    with args.problems_path.open("rb") as f:
        pset_dict = pkl.load(f)
    problems = pset_dict["problems"][0]  # todo since no curriclulums

    problems_indexer = mp.Value("I", 0)
    problems_indices = mp.Array("I", range(len(problems)))
    problems_loader = AsyncProblemLoader(
        problems, problems_indices, problems_indexer, args.seed
    )

    if args.mode == "train" and args.valid_path:
        with args.valid_path.open("rb") as f:
            vset_dict = pkl.load(f)
        valid_problems = vset_dict["problems"][0]  # todo since no curriculums
        valid_problems_indexer = mp.Value("I", 0)
        valid_indices = mp.Array("I", range(len(problems)))
        valid_problems_loader = AsyncProblemLoader(
            valid_problems, valid_indices, valid_problems_indexer, args.seed
        )
    else:
        valid_problems = None
    results_queue = mp.Queue()

    if args.mode == "train":
        if args.checkpoint_path is None:
            if "SLURM_JOB_ID" in os.environ:
                runid = (
                    f"{os.environ['SLURM_JOB_ID']}-{os.environ['SLURM_ARRAY_TASK_ID']}"
                )
            else:
                runid = f"{int(abs_start_time)}"
            run_name = f"{args.problems_path.parent.stem}-{args.problems_path.stem}_{args.agent}{exp_name}_{args.seed}_{runid}"
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
        logdir /= f"test_{model_name}{exp_name}_{args.seed}_{int(abs_start_time)}"
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    logdir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to {str(logdir)}")

    # Witness domains don't support conditional backward search
    if pset_dict["domain_name"] != "Witness" and (not args.no_conditional_backward):
        args.conditional_backward = True
    else:
        args.conditional_backward = False

    dummy_domain = deepcopy(pset_dict["problems"][0][0].domain)
    num_raw_features = dummy_domain.state_tensor(dummy_domain.init()).size().numel()
    derived_args = {
        "conditional_backward": args.conditional_backward,
        "state_t_width": dummy_domain.state_t_width,
        "state_t_depth": dummy_domain.state_t_depth,
        "num_actions": dummy_domain.num_actions,
        "in_channels": dummy_domain.in_channels,
        "num_raw_features": num_raw_features,
    }

    agent_class = getattr(sa, args.agent)
    agent = agent_class(logdir, args, derived_args)

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
    if args.test_expansion_budget < 0:
        args.test_expansion_budget = args.train_expansion_budget
    if args.max_expansion_budget < 0:
        args.max_expansion_budget = args.train_expansion_budget

    if args.master_port == "auto":
        os.environ["MASTER_PORT"] = find_free_port(args.lockfile, args.master_addr)
    else:
        os.environ["MASTER_PORT"] = args.master_port

    print(f"Using port {os.environ['MASTER_PORT']}")
    os.environ["MASTER_ADDR"] = args.master_addr

    processes = []
    for rank in range(args.world_size):
        proc = mp.Process(
            target=run,
            args=(
                rank,
                agent,
                args,
                problems,
                valid_problems,
                results_queue,
            ),
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()
