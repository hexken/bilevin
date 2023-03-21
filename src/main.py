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

import argparse
import json
import os
from pathlib import Path
import random
import socket
import time
from typing import Optional

import numpy as np
from test import test
import torch as to
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
import wandb

import domains
from loaders import CurriculumLoader, ProblemsBatchLoader
from models import ConvNetDouble, ConvNetSingle
import models.loss_functions as loss_fns
from search import BiBS, BiLevin, Levin
from search.agent import Agent
from train import train


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--problemset-path",
        type=lambda p: Path(p).absolute(),
        help="path of file with problem instances",
    )
    parser.add_argument(
        "-v",
        "--validset-path",
        type=lambda p: Path(p).absolute(),
        help="path of file with problem instances",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=lambda p: Path(p).absolute(),
        default=None,
        help="path of directory to load previously saved model(s) from",
    )
    parser.add_argument(
        "--model-suffix",
        type=str,
        default="best",
        help="suffix of model to load, i.e. forward_[suffix].pt",
    )
    parser.add_argument(
        "-l",
        "--loss-fn",
        type=str,
        default="levin_loss",
        choices=[
            "levin_loss",
            "levin_loss_avg",
            "cross_entropy_loss",
        ],
        help="loss function",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.001,
        help="l2 regularization weight",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
        help="optimizer learning rate",
    )
    parser.add_argument(
        "-g",
        "--grad-steps",
        type=int,
        default=10,
        help="number of gradient steps to be performed in each opt pass",
    )
    parser.add_argument(
        "--bootstrap-epochs",
        type=int,
        default=1,
        help="number of bootstrap epochs to train for",
    )
    parser.add_argument(
        "--curriculum-epochs",
        type=int,
        default=1,
        help="number of curriculum epochs to train for",
    )
    parser.add_argument(
        "--permutation-epochs",
        type=int,
        default=1,
        help="number of permutation epochs to train for",
    )
    parser.add_argument(
        "-r",
        "--epochs-reduce-lr",
        type=int,
        default=99999,
        help="reduce learning rate by a factor of 10 after this many epochs",
    )
    parser.add_argument(
        "--epoch-begin-validate",
        type=int,
        default=1,
        help="reduce learning rate by a factor of 10 after this many epochs",
    )
    parser.add_argument(
        "--shuffle-trajectory",
        action="store_true",
        help="shuffle trajectory states",
    )
    parser.add_argument(
        "-a",
        "--agent",
        type=str,
        choices=["Levin", "BiLevin", "BiBS"],
        help="name of the search agent",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="number of processes to spawn",
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        default=socket.gethostname(),
        help="address for multiprocessing communication",
    )
    parser.add_argument(
        "--master-port",
        type=str,
        default="34567",
        help="port for multiprocessing communication",
    )
    parser.add_argument(
        "--batch-size-train",
        type=int,
        default=32,
        help="number of problems to batch during",
    )
    parser.add_argument(
        "--batch-size-print",
        type=int,
        default=None,
        help="number of results to print per block during testint",
    )
    parser.add_argument(
        "--initial-budget",
        type=int,
        default=2**10,
        help="initial budget (nodes expanded) to solve a problem",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="train or test the model from model-folder using instances from problems-folder",
    )
    parser.add_argument(
        "--exp-name", type=str, default="", help="the name of this experiment"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed of the experiment",
    )
    parser.add_argument(
        "--torch-deterministic",
        action="store_true",
        help="set `torch.backends.cudnn.deterministic=False` and `torch.use_deterministic_agents(True)`",
    )
    parser.add_argument(
        "--update-levin-costs",
        action="store_true",
        help="update levin costs when cheaper path found",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="disabled",
        choices=["disabled", "online", "offline"],
        help="track with Weights and Biases",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="bilevin",
        help="the wandb's project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="ken-levi",
        help="the entity (team) of the wandb project",
    )
    args = parser.parse_args()
    return args


def run(rank, run_name, model_args, args, local_loader, local_valid_loader):
    is_distributed = args.world_size > 1

    if is_distributed:
        os.environ["MASTER_ADDR"] = args.master_addr
        os.environ["MASTER_PORT"] = args.master_port
        dist.init_process_group(backend="gloo", rank=rank, world_size=args.world_size)

    if args.mode == "test":
        run_name = f"test_{run_name}"

    if rank == 0:
        wandb.init(
            mode=args.wandb_mode,
            project=args.wandb_project,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
            settings=wandb.Settings(code_dir="src/"),
        )
        if args.wandb_mode != "disabled":
            print(
                f"Logging with Weights and Biases\n  to {args.wandb_entity}/{args.wandb_project}/{run_name}"
            )

        print(f"Logging with tensorboard\n  to runs/{run_name}\n")

        writer = SummaryWriter(f"runs/{run_name}")
        arg_string = "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        )
        for arg in arg_string.splitlines()[2:]:
            arg = arg.replace("|", "", 1)
            arg = arg.replace("|", ": ", 1)
            arg = arg.replace("|", "", 1)
            print(arg)
        print()

        writer.add_text(
            "hyperparameters",
            arg_string,
        )

    else:
        writer = None

    local_seed = args.seed + rank
    random.seed(local_seed)
    np.random.seed(local_seed)
    to.manual_seed(local_seed)
    if args.torch_deterministic:
        to.use_deterministic_algorithms(True)
        to.backends.cudnn.benchmark = False  # type:ignore
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    model = None
    forward_model = None
    backward_model = None
    if args.agent == "Levin":
        agent = Levin()
    elif args.agent == "BiLevin":
        agent = BiLevin()
    elif args.agent == "BiBS":
        agent = BiBS()
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    if args.agent == "Levin":
        forward_model = ConvNetSingle(
            model_args["in_channels"],
            model_args["state_t_width"],
            (2, 2),
            32,
            model_args["num_actions"],
        )
    elif args.agent == "BiLevin":
        forward_model = ConvNetSingle(
            model_args["in_channels"],
            model_args["state_t_width"],
            (2, 2),
            32,
            model_args["num_actions"],
        )
        if model_args["double_backward"]:
            backward_model = ConvNetDouble(
                model_args["in_channels"],
                model_args["state_t_width"],
                (2, 2),
                32,
                model_args["num_actions"],
            )
        else:
            backward_model = ConvNetSingle(
                model_args["in_channels"],
                model_args["state_t_width"],
                (2, 2),
                32,
                model_args["num_actions"],
            )

    if agent.trainable:
        if agent.bidirectional:
            assert isinstance(forward_model, nn.Module)
            assert isinstance(backward_model, nn.Module)
            model = forward_model, backward_model
        else:
            assert isinstance(forward_model, nn.Module)
            model = forward_model

        if args.model_path is None:
            # just use the random initialization from rank 0
            if is_distributed:
                for param in forward_model.parameters():
                    dist.broadcast(param.data, 0)
                if agent.bidirectional:
                    assert isinstance(backward_model, nn.Module)
                    for param in backward_model.parameters():
                        dist.broadcast(param.data, 0)
        elif args.model_path.is_dir():
            forward_model_path = args.model_path / f"forward_{args.model_suffix}.pt"
            forward_model.load_state_dict(to.load(forward_model_path))
            if agent.bidirectional:
                assert isinstance(backward_model, nn.Module)
                backward_model_path = (
                    args.model_path / f"backward_{args.model_suffix}.pt"
                )
                backward_model.load_state_dict(to.load(backward_model_path))

            if rank == 0:
                print(f"Loaded model(s)\n  from  {str(args.model_path)}")
        else:
            raise ValueError("model-path argument must be a directory if given")

    model_save_path = Path(__file__).parent.parent / f"runs/{run_name}"
    model_save_path.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        if args.mode == "train":
            print(f"Saving model(s)\n  to {str(model_save_path)}")
        elif args.mode == "test" and agent.trainable:
            assert isinstance(forward_model, nn.Module)
            to.save(forward_model.state_dict(), model_save_path / "forward.pt")
            if agent.bidirectional:
                assert isinstance(backward_model, nn.Module)
                to.save(backward_model.state_dict(), model_save_path / "backward.pt")

            print(f"Copied model(s) to use\n  to {str(model_save_path)}")

    if args.mode == "train":
        assert model
        loss_fn = getattr(loss_fns, args.loss_fn)
        optimizer_cons = to.optim.Adam
        optimizer_params = {
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        }

        train(
            rank,
            agent,
            model,
            model_save_path,
            loss_fn,
            optimizer_cons,
            optimizer_params,
            local_loader,
            writer,
            args.world_size,
            args.update_levin_costs,
            budget=args.initial_budget,
            seed=local_seed,
            grad_steps=args.grad_steps,
            epochs_reduce_lr=args.epochs_reduce_lr,
            epoch_begin_validate=args.epoch_begin_validate,
            shuffle_trajectory=args.shuffle_trajectory,
            valid_loader=local_valid_loader,
        )

    elif args.mode == "test":
        test(
            rank,
            agent,
            model,
            local_loader,
            writer,
            args.world_size,
            update_levin_costs=args.update_levin_costs,
            initial_budget=args.initial_budget,
            increase_budget=True,
            print_results=True,
            validate=False,
            epoch=None,
        )

    if rank == 0:
        total_time = time.time() - start_time
        print(f"Total time: {total_time:.2f} seconds")
        writer.add_text("total_time", f"{total_time:.2f} seconds")
        writer.close()
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()

    is_distributed = args.world_size > 1

    if args.mode == "train":
        if args.batch_size_train < args.world_size:
            raise ValueError(
                f"batch-size-train'{args.batch_size_train}' must be >= world_size {args.world_size}"
            )
        if args.batch_size_train % args.world_size != 0:
            raise ValueError(
                f"batch-size-train '{args.batch_size_train}' must be a multiple of world_size {args.world_size}"
            )

    start_time = time.time()

    problemset_dict = json.load(args.problemset_path.open("r"))
    domain_module = getattr(domains, problemset_dict["domain_module"])
    problemset = getattr(domain_module, "parse_problemset")(problemset_dict)

    # problems_per_difficulty = problemset["problems_per_difficulty"]
    # if problems_per_difficulty % args.world_size != 0:
    #     raise ValueError("problems_per_difficulty must be a multiple of world_size")

    if args.validset_path:
        validset_dict = json.load(args.validset_path.open("r"))
        validset = getattr(domain_module, "parse_problemset")(validset_dict)

    print(time.ctime(start_time))

    def get_loaders(problemset):
        def split(problems):
            num_problems_parsed = len(problems)
            if num_problems_parsed < args.world_size:
                raise Exception(
                    f"Number of problems '{num_problems_parsed}' must be greater than world size '{args.world_size}'"
                )

            problemsets = [[] for _ in range(args.world_size)]
            proc = 0
            for problem in problems:
                # this should be a redundant check, but just in case
                if problem.domain.is_goal(problem.domain.initial_state):
                    raise Exception(
                        f"Problem '{problem.id}' initial state is a goal state"
                    )

                problemsets[proc].append(problem)
                proc = (proc + 1) % args.world_size

            print(f"Parsed {num_problems_parsed} problems")

            large_size = len(problemsets[0])
            small_size = len(problemsets[-1])
            if large_size == small_size:
                print(
                    f"  Loading {large_size} into each of {args.world_size} processes"
                )
            else:
                small_ranks = 0
                while len(problemsets[small_ranks]) == large_size:
                    small_ranks += 1
                    continue

                print(
                    f"  Loading {large_size} into ranks 0-{small_ranks - 1},\n"
                    f"          {small_size} into ranks {small_ranks}-{args.world_size - 1}\n"
                )

            return problemsets, large_size

        local_batch_size = args.batch_size_train // args.world_size

        def set_id_idxs(start_idx, problems):
            for i, p in enumerate(
                problems,
                start=start_idx,
            ):
                p.id_idx = i

        if "is_curriculum" in problemset:
            # for now, all training problemsets should be curricula
            bootstrap_problemsets, bs_large_size = split(
                problemset["bootstrap_problems"]
            )
            all_bootstrap_ids = [p.id for p in problemset["bootstrap_problems"]]
            set_id_idxs(0, problemset["bootstrap_problems"])

            curriculum_problems = problemset["curriculum_problems"]
            all_curr_ids = [p.id for p in problemset["curriculum_problems"]]
            set_id_idxs(len(all_bootstrap_ids), curriculum_problems)
            ppd = problemset["problems_per_difficulty"]
            num_difficulty_levels = len(problemset["curriculum"])

            curriculum_diff_ranks_split = [[] for _ in range(num_difficulty_levels)]
            for i in range(num_difficulty_levels):
                curriculum_difficulty_problems = curriculum_problems[
                    i * ppd : (i + 1) * ppd
                ]
                curriculum_diff_ranks_split[i], curr_large_size = split(
                    curriculum_difficulty_problems
                )
            curr_large_size *= num_difficulty_levels  # assumes all curriculum difficulties are the same size

            curriculum_problemsets = [[] for _ in range(args.world_size)]
            for i in range(args.world_size):
                for j in range(num_difficulty_levels):
                    curriculum_problemsets[i].extend(curriculum_diff_ranks_split[j][i])

            # for pset in curriculum_problemsets:
            #     print([p.id for p in pset])

            permutation_problemsets, perm_large_size = split(
                problemset["permutation_problems"]
            )
            all_permutation_ids = [p.id for p in problemset["permutation_problems"]]
            set_id_idxs(
                len(all_bootstrap_ids) + len(all_curr_ids),
                problemset["permutation_problems"],
            )

            loaders = []
            for rank in range(args.world_size):

                loaders.append(
                    CurriculumLoader(
                        bootstrap_problems=bootstrap_problemsets[rank],
                        all_bootstrap_ids=all_bootstrap_ids,
                        bootstrap_epochs=args.bootstrap_epochs,
                        curriculum=problemset["curriculum"],
                        problems_per_difficulty=ppd // args.world_size,
                        curriculum_problems=curriculum_problemsets[rank],
                        all_curriculum_ids=all_curr_ids,
                        curriculum_epochs=args.curriculum_epochs,
                        permutation_problems=permutation_problemsets[rank],
                        all_permutation_ids=all_permutation_ids,
                        permutation_epochs=args.permutation_epochs,
                        batch_size=local_batch_size,
                        world_size=args.world_size,
                        seed=args.seed + rank,
                    )
                )

        else:
            # this is only for loading test/valid problemsets, which always use a batch_size of 1 to
            # populate lists/tuples inside the test script
            loaders = []
            problemsets, N = split(problemset["problems"])
            all_ids = [p.id for p in problemset["problems"]]
            set_id_idxs(0, problemset["problems"])

            for rank in range(args.world_size):
                loaders.append(
                    ProblemsBatchLoader(
                        problems=problemsets[rank],
                        all_ids=all_ids,
                        batch_size=1,
                        world_size=args.world_size,
                        seed=args.seed,
                    )
                )

        return loaders

    problem_loaders = get_loaders(problemset)

    valid_loaders = None
    if args.validset_path:
        valid_loaders = get_loaders(validset)

    exp_name = f"_{args.exp_name}" if args.exp_name else ""
    problemset_params = (
        f"{args.problemset_path.parent.stem}-{args.problemset_path.stem}"
    )
    run_name = f"{problemset_dict['domain_name']}-{problemset_params}_{args.agent}-{args.initial_budget}{exp_name}_{args.seed}_{int(start_time)}"
    del problemset_dict

    model_args = {
        "in_channels": problemset["in_channels"],
        "state_t_width": problemset["state_t_width"],
        "num_actions": problemset["num_actions"],
        "double_backward": problemset["double_backward"],
    }

    if is_distributed:
        processes = []
        for rank in range(args.world_size):
            proc = mp.Process(
                target=run,
                args=(
                    rank,
                    run_name,
                    model_args,
                    args,
                    problem_loaders[rank],
                    valid_loaders[rank] if valid_loaders else None,
                ),
            )
            problem_loaders[rank] = None
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()
    else:
        assert len(problem_loaders) == 1
        run(
            0,
            run_name,
            model_args,
            args,
            problem_loaders[0],
            valid_loaders[0] if valid_loaders else None,
        )
