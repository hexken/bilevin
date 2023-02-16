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
        "-e",
        "--epohcs",
        type=int,
        default=10,
        help="number of epochs to train for",
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
        "--time-limit-overall",
        type=int,
        default="6000",
        help="time limit in seconds for solving whole problem set",
    )
    parser.add_argument(
        "--time-limit-each",
        type=int,
        default="300",
        help="time limit in seconds for solving each problem",
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


def run(rank, run_name, model_args, args, problemset, queue, validset):
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
        writer = SummaryWriter()
        writer.close()

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
            forward_model_path = args.model_path / "forward.pt"
            forward_model.load_state_dict(to.load(forward_model_path))
            if agent.bidirectional:
                assert isinstance(backward_model, nn.Module)
                backward_model_path = args.model_path / "backward.pt"
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

        local_batch_size = args.batch_size_train // args.world_size

        train(
            rank,
            agent,
            model,
            model_save_path,
            loss_fn,
            optimizer_cons,
            optimizer_params,
            problemset,
            local_batch_size,
            writer,
            args.world_size,
            args.update_levin_costs,
            budget=args.initial_budget,
            seed=local_seed,
            grad_steps=args.grad_steps,
            shuffle_trajectory=args.shuffle_trajectory,
            valid_problems=validset,
            results_queue=queue,
        )

    elif args.mode == "test":
        test(
            agent,
            model,
            problemset,
            writer,
            args.world_size,
            False,
            args.initial_budget,
            queue,
            args.batch_size_print,
        )
        queue.close()

    if rank == 0:
        total_time = time.time() - start_time
        print(f"Total time: {total_time:.2f} seconds")
        writer.add_text("total_time", f"{total_time:.2f} seconds")
        writer.close()


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
    (
        parsed_problems,
        num_actions,
        in_channels,
        state_t_width,
        double_backward,
    ) = getattr(domain_module, "load_problemset")(problemset_dict)

    if args.validset_path:
        validset_dict = json.load(args.validset_path.open("r"))
        (valid_parsed_problems, _, _, _, _,) = getattr(
            domain_module, "load_problemset"
        )(validset_dict)

    def split_problems(problems):
        num_problems_parsed = len(parsed_problems)
        if num_problems_parsed < args.world_size:
            raise Exception(
                f"Number of problems '{num_problems_parsed}' must be greater than world size '{args.world_size}'"
            )

        for p in parsed_problems:
            if p.domain.is_goal(p.domain.initial_state):
                raise Exception(f"Problem '{p.id}' initial state is a goal state")

        problems_per_process = num_problems_parsed // args.world_size
        problemsets = []
        for proc in range(args.world_size):
            problems_local = parsed_problems[
                proc * problems_per_process : (proc + 1) * problems_per_process
            ]
            problemsets.append(problems_local)

        num_remaining_problems = num_problems_parsed - (
            problems_per_process * args.world_size
        )
        if num_remaining_problems > 0:
            for i, problem in enumerate(parsed_problems[-num_remaining_problems:]):
                problemsets[i].append(problem)

        print(time.ctime(start_time))
        print(f"Parsed {num_problems_parsed} problems")
        if len(problemsets[0]) == len(problemsets[-1]):
            print(
                f"  Loading {problems_per_process} into each of {args.world_size} processes\n"
            )
        else:
            print(
                f"  Loading {len(problemsets[0])} into ranks 0-{num_remaining_problems - 1},\n"
                f"          {problems_per_process} into ranks {num_remaining_problems}-{args.world_size - 1}\n"
            )
        return problemsets

    problemsets = split_problems(parsed_problems)

    validsets = None
    if args.validset_path:
        validsets = split_problems(valid_parsed_problems)

    exp_name = f"_{args.exp_name}" if args.exp_name else ""
    problemset_params = (
        f"{args.problemset_path.parent.stem}-{args.problemset_path.stem}"
    )
    run_name = f"{problemset_dict['domain_name']}-{problemset_params}_{args.agent}-{args.initial_budget}{exp_name}_{args.seed}_{int(start_time)}"
    del problemset_dict

    model_args = {
        "in_channels": in_channels,
        "state_t_width": state_t_width,
        "num_actions": num_actions,
        "double_backward": double_backward,
    }
    queue = None
    if args.mode == "test" or args.validset_path:
        queue = mp.Queue()

    if is_distributed:
        processes = []
        for rank in range(args.world_size):
            p = mp.Process(
                target=run,
                args=(
                    rank,
                    run_name,
                    model_args,
                    args,
                    problemsets[rank],
                    queue,
                    validsets[rank] if validsets else None,
                ),
            )
            problemsets[rank] = None
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        assert len(problemsets) == 1
        run(
            0,
            run_name,
            model_args,
            args,
            problemsets[0],
            queue,
            validsets[0] if validsets else None,
        )
