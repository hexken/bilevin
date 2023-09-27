# Copyright (C) 2021-2022, Ken Tjhia
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import json
import json
import os
from pathlib import Path
import random
import time
from timeit import default_timer as timer
from typing import Optional

import numpy as np
from test import test
import torch as to
import torch.distributed as dist
import torch.multiprocessing as mp

from args import parse_args
import domains
from search.bilevin import BiLevin
from search.levin import Levin
from train import train

# todo add args for kernel dims


def split(args, problemset):
    "split a list of lists of problems into a list of lists of problems per rank"
    rng = np.random.default_rng(args.seed)

    def _split(problems):
        ranks_problems = [[] for _ in range(args.world_size)]
        rank = 0
        for problem in problems:
            # this should be a redundant check, but just in case
            if problem.domain.is_goal(problem.domain.initial_state):
                raise Exception(f"Problem '{problem.id}' initial state is a goal state")

            ranks_problems[rank].append(problem)
            rank = (rank + 1) % args.world_size

        # ensure all ranks have same number of problems per stage
        n_largest_pset = len(ranks_problems[0])
        for pset in ranks_problems:
            if len(pset) < n_largest_pset:
                pset.append(rng.choice(problems))
            assert len(pset) == n_largest_pset

        return ranks_problems

    def set_id_idxs(problems):
        for i, p in enumerate(problems):
            p.id_idx = i

    stages_x_problems = problemset["problems"]
    num_stages = len(stages_x_problems)
    all_ids = [p.id for problems in stages_x_problems for p in problems]

    # turn stages x problems into stages x ranks x problems
    stages_x_ranks_x_problems = [[] for _ in range(num_stages)]
    for stage in range(num_stages):
        set_id_idxs(stages_x_problems[stage])
        stages_x_ranks_x_problems[stage] = _split(stages_x_problems[stage])

    ranks_x_stages_x_problems = [[] for _ in range(args.world_size)]
    for rank in range(args.world_size):
        for stage in range(num_stages):
            ranks_x_stages_x_problems.append(stages_x_ranks_x_problems[stage][rank])

    return ranks_x_stages_x_problems, all_ids


def run(
    rank,
    run_name,
    model_args,
    args,
    local_train_problems,
    local_train_ids,
    local_valid_problems,
    local_valid_ids,
):
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    dist.init_process_group(backend="gloo", rank=rank, world_size=args.world_size)

    if args.mode == "test":
        run_name = f"test_{run_name}"

    local_seed = args.seed + rank
    random.seed(local_seed)
    np.random.seed(local_seed)
    to.manual_seed(local_seed)

    if args.agent == "Levin":
        agent = Levin(rank, logdir, args, model_args)
    elif args.agent == "BiLevin":
        agent = BiLevin(rank, logdir, args, model_args)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    if args.mode == "train":
        train(
            args,
            rank,
            agent,
            local_train_problems,
            local_train_ids,
            local_valid_problems,
            local_valid_ids,
            seed=local_seed,
        )

    elif args.mode == "test":
        test(
            args,
            rank,
            agent,
            local_train_problems,
            seed=local_seed,
            print_results=True,
            validate=False,
            epoch=None,
        )

    if rank == 0:
        total_time = timer() - rel_start_time
        print(f"Total time: {total_time:.2f} seconds")
        with (logdir / "total_time_seconds.txt").open("w") as f:
            f.write(f"{total_time:.2f}")


if __name__ == "__main__":
    args = parse_args()

    abs_start_time = time.time()
    rel_start_time = timer()
    print(time.ctime(abs_start_time))

    problemset_dict = json.load(args.problemset_path.open("r"))
    domain_module = getattr(domains, problemset_dict["domain_module"])
    problemset, model_args = getattr(domain_module, "parse_problemset")(problemset_dict)
    train_problems, train_ids = split(args, problemset)

    if args.validset_path:
        validset_dict = json.load(args.validset_path.open("r"))
        validset, _ = getattr(domain_module, "parse_problemset")(validset_dict)
        valid_problems, valid_ids = split(args, validset)
    else:
        valid_problems = None
        valid_ids = None

    exp_name = f"_{args.exp_name}" if args.exp_name else ""
    problemset_params = (
        f"{args.problemset_path.parent.stem}-{args.problemset_path.stem}"
    )
    run_name = f"{problemset_dict['domain_name']}-{problemset_params}_{args.agent}-e{args.expansion_budget}-t{args.time_budget}{exp_name}_{args.seed}_{int(abs_start_time)}"
    del problemset_dict

    model_args.update(
        {
            "kernel_size": (2, 2),
            "num_filters": 32,
            "share_feature_net": args.share_feature_net,
            "forward_hidden_layers": args.forward_hidden_layers,
            "backward_hidden_layers": args.backward_hidden_layers,
        }
    )
    model_args["backward_goal"] = (
        model_args["requires_backward_goal"] and not args.no_backward_goal
    )

    logdir = args.runsdir_path / run_name
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to {str(logdir)}\n")

    arg_dict = {
        k: (v if not isinstance(v, Path) else str(v)) for k, v in vars(args).items()
    }
    with (logdir / "args.json").open("w") as f:
        json.dump(arg_dict, f, indent=2)

    args.logdir = logdir

    processes = []
    for rank in range(args.world_size):
        proc = mp.Process(
            target=run,
            args=(
                rank,
                run_name,
                model_args,
                args,
                train_problems[rank],
                train_ids[rank],
                valid_problems[rank] if valid_problems else None,
                valid_ids[rank] if valid_ids else None,
            ),
        )
        proc.start()
        processes.append(proc)

    del train_problems
    del valid_problems

    for proc in processes:
        proc.join()
