import datetime
import json
import os
from pathlib import Path
import pickle as pkl
import time
from timeit import default_timer as timer

from filelock import FileLock, Timeout
import numpy as np
import torch as to
import torch.distributed as dist
import torch.multiprocessing as mp

from args import parse_args
from loaders import ProblemLoader
from search.astar import AStar, BiAStarAlt, BiAStarBFS
from search.levin import BiLevinAlt, BiLevinBFS, Levin
from search.phs import BiPHSAlt, BiPHSBFS, PHS
from search.utils import set_seeds
from test import test
from train import train
from utils import find_free_port


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
    for stage_problems in stages_x_problems:
        stages_x_ranks_x_problems.append(split_by_rank(stage_problems))

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
            print(
                f"\nTraining on {len(local_problems)} stages, {args.n_final_stage_epochs} epochs for final stage"
            )
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

    with args.problems_path.open("rb") as f:
        pset_dict = pkl.load(f)
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

    # Witness domains don't support conditional backward search
    conditional_backward = (
        args.conditional_backward and pset_dict["domain_name"] != "Witness"
    )
    dummy_domain = pset_dict["problems"][0][0].domain
    num_features = dummy_domain.state_tensor(dummy_domain.reset()).size().numel()
    aux_args = {
        "conditional_backward": conditional_backward,
        "state_t_width": dummy_domain.state_t_width,
        "state_t_depth": dummy_domain.state_t_depth,
        "num_actions": dummy_domain.num_actions,
        "in_channels": dummy_domain.in_channels,
        "num_features": num_features,
    }

    agent_class = globals()[args.agent]
    agent = agent_class(logdir, args, aux_args)

    if args.valid_path:
        with args.valid_path.open("rb") as f:
            vset_dict = pkl.load(f)
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
    if args.test_expansion_budget < 0:
        args.test_expansion_budget = args.train_expansion_budget
    if args.max_expansion_budget < 0:
        args.max_expansion_budget = args.train_expansion_budget

    if args.master_port == "auto":
        lockfile = f"{args.lockfile}.lock"
        portfile = Path(f"{args.lockfile}.pkl")
        lock = FileLock(lockfile)
        with lock:
            if portfile.is_file():
                f = portfile.open("r+b")
                ports = pkl.load(f)
                while True:
                    port = find_free_port(args.master_addr)
                    if port in ports:
                        continue
                    else:
                        break
                f.seek(0)
                ports.add(port)
                pkl.dump(ports, f)
                f.truncate()
                f.close()
            else:
                port = find_free_port(args.master_addr)
                ports = {port}
                with portfile.open("wb") as f:
                    pkl.dump(ports, f)
        os.environ["MASTER_PORT"] = str(port)
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
