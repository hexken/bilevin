from argparse import Namespace
from copy import deepcopy
import datetime
import gc
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
from search.loaders import ArrayLoader
from search.utils import print_search_summary
from test import test
from train import train
from utils import find_free_port, set_seeds


def run(
    rank: int,
    agent: Agent,
    args: Namespace,
    train_loader: ArrayLoader,
    valid_loader: ArrayLoader,
    test_loader: ArrayLoader | None,
    results_queue: mp.Queue,
):
    dist.init_process_group(
        timeout=datetime.timedelta(seconds=86400),
        backend="gloo",
        rank=rank,
        world_size=args.world_size,
    )

    gc.collect()
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
            results_queue,
        )
        if rank == 0:
            (args.logdir / "training_completed.txt").open("w").close()

    ts = time.time()
    if test_loader is not None:
        if rank == 0:
            if args.mode == "train":
                model_path = agent.logdir / "model_best_expanded.pt"
                agent.load_model(model_path)
                print(f"Testing using model {model_path.name}")
            else:
                print("\nTesting...")

        results_df = test(
            rank,
            agent,
            test_loader,
            results_queue,
            args.test_expansion_budget,
            args.time_budget,
            print_results=True,
            solved_results_path=args.logdir / f"solved_results.pkl",
            results_df_path=args.logdir / f"test.pkl",
        )
        if rank == 0:
            print(f"Testing took {time.time() - ts:.2f} seconds")
            print_search_summary(results_df)
    dist.barrier()


if __name__ == "__main__":
    # mp.set_start_method("fork")
    abs_start_time = time.time()
    print(time.ctime(abs_start_time))

    args = parse_args()
    if args.master_lr > 0:
        args.forward_feature_net_lr = args.master_lr
        args.backward_feature_net_lr = args.master_lr
        args.forward_policy_lr = args.master_lr
        args.backward_policy_lr = args.master_lr
        args.forward_heuristic_lr = args.master_lr
        args.backward_heuristic_lr = args.master_lr

    to.set_num_threads(1)
    to.set_default_dtype(to.float64)
    set_seeds(args.seed)

    exp_name = f"_{args.exp_name}" if args.exp_name else ""

    if args.test_path is not None:
        test_loader, pset_dict = ArrayLoader.from_path(args, args.test_path)
        test_loader_len = len(test_loader)
    else:
        test_loader = None
        test_loader_len = 0

    model = None
    if args.mode == "train":
        train_loader, pset_dict = ArrayLoader.from_path(args, args.train_path)
        valid_loader, _ = ArrayLoader.from_path(args, args.valid_path)
        test_loader_len = max(test_loader_len, args.batch_size, len(valid_loader))
        # SLURM_ARRAY_JOB_ID
        if args.checkpoint_path is None:
            if "SLURM_ARRAY_JOB_ID" in os.environ:
                runid = f"{os.environ['SLURM_ARRAY_JOB_ID']}-{os.environ['SLURM_ARRAY_TASK_ID']}-{os.environ['SLURM_JOB_ID']}"
            elif "SLURM_JOB_ID" in os.environ:
                runid = f"{os.environ['SLURM_JOB_ID']}"
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
        else:
            raise ValueError("Must specify model_path to test")
        logdir /= f"test_{model_name}{exp_name}_{args.seed}_{int(abs_start_time)}"

    logdir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to {str(logdir)}")

    assert pset_dict
    domain = deepcopy(pset_dict["problems"][0].domain)
    num_raw_features = domain.state_tensor(domain.init()).size().numel()
    derived_args = {
        "conditional_forward": pset_dict["conditional_forward"],
        "conditional_backward": pset_dict["conditional_backward"],
        "state_t_width": domain.state_t_width,
        "state_t_depth": domain.state_t_depth,
        "num_actions": domain.num_actions,
        "in_channels": domain.in_channels,
        "num_raw_features": num_raw_features,
    }

    agent_class = getattr(sa, args.agent)
    agent = agent_class(logdir, args, derived_args)
    if args.mode == "test":
        agent.load_model(args.model_path)

    arg_dict = {
        k: (v if not isinstance(v, Path) else str(v)) for k, v in vars(args).items()
    }
    argspath = logdir / "args.json"
    if not argspath.is_file():
        with argspath.open("w") as f:
            json.dump(arg_dict, f, indent=2)
    for k, v in arg_dict.items():
        print(f"{k}: {v}")
    del arg_dict, domain, argspath, pset_dict

    args.logdir = logdir
    if args.test_expansion_budget < 0:
        args.test_expansion_budget = args.train_expansion_budget

    if args.master_port == "auto":
        os.environ["MASTER_PORT"] = find_free_port(args.lockfile, args.master_addr)
    else:
        os.environ["MASTER_PORT"] = args.master_port

    print(f"Using port {os.environ['MASTER_PORT']}")
    os.environ["MASTER_ADDR"] = args.master_addr

    results_queue = mp.Queue(maxsize=test_loader_len)
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
