from argparse import Namespace
from copy import deepcopy
import datetime
import json
import os
from pathlib import Path
import pickle as pkl
import time
from timeit import default_timer as timer

import torch as to
import torch.distributed as dist
import torch.multiprocessing as mp

from args import parse_args
from search.agent import Agent
import search.agents as sa
from search.loaders import AsyncProblemLoader
from search.utils import print_search_summary, set_seeds
from test import test
from train import train
from utils import find_free_port, get_loader


def run(
    rank: int,
    agent: Agent,
    args: Namespace,
    train_loader: AsyncProblemLoader,
    valid_loader: AsyncProblemLoader,
    test_loader: AsyncProblemLoader | None,
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
            print(
                f"\nTraining on {len(train_loader)} problems for {args.n_epochs} epochs"
            )
        train(
            args,
            rank,
            agent,
            train_loader,
            valid_loader,
            test_loader,
            results_queue,
        )
        (args.logdir / "training_completed.txt").open("w").close()

    if test_loader is not None:
        if rank == 0:
            print("\nTesting...")
        results_df = test(
            args,
            rank,
            agent,
            test_loader,
            results_queue,
            print_results=True,
        )
        if rank == 0:
            with open(args.logdir / f"test.pkl", "wb") as f:
                pkl.dump(results_df, f)
            print_search_summary(results_df, bidirectional=agent.is_bidirectional)


if __name__ == "__main__":
    # mp.set_start_method("fork")
    args = parse_args()
    abs_start_time = time.time()
    rel_start_time = timer()
    print(time.ctime(abs_start_time))
    set_seeds(args.seed)
    exp_name = f"_{args.exp_name}" if args.exp_name else ""

    results_queue = mp.Queue()

    if args.test_path is not None:
        test_loader, pset_dict = get_loader(args, args.test_path)
    else:
        test_loader = None

    if args.mode == "train":
        train_loader, pset_dict = get_loader(
            args, args.train_path, batch_size=args.batch_size
        )
        valid_loader, _ = get_loader(args, args.valid_path)
        if args.checkpoint_path is None:
            if "SLURM_JOB_ID" in os.environ:
                runid = (
                    f"{os.environ['SLURM_JOB_ID']}-{os.environ['SLURM_ARRAY_TASK_ID']}"
                )
            else:
                runid = f"{int(abs_start_time)}"
            run_name = f"{args.train_path.parent.stem}-{args.train_path.stem}_{args.agent}{exp_name}_{args.seed}_{runid}"
            logdir = args.runsdir_path / run_name
        else:
            run_name = str(args.checkpoint_path.parent)
            logdir = args.checkpoint_path.parent
            print(f"Loaded checkpoint {str(args.checkpoint_path)}")

    else:  # test
        train_loader = valid_loader = None
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
                train_loader,
                valid_loader,
                test_loader,
                results_queue,
            ),
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()
