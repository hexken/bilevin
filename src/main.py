import argparse
import json
import os
from pathlib import Path
import random
import time
from typing import Optional

import numpy as np
import torch as to
import torch.distributed as dist
from torch.utils.tensorboard.writer import SummaryWriter
import wandb

import domains
from models import ConvNetDouble, ConvNetSingle
import models.loss_functions as loss_fns
from search import BiLevin, Levin
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
        "-m",
        "--model-path",
        type=lambda p: Path(p).absolute(),
        default=Path(__file__).parent / "trained_models",
        help="path of file to load or directory to save model",
    )
    parser.add_argument(
        "-l",
        "--loss-fn",
        type=str,
        default="levin_loss_avg",
        choices=[
            "levin_loss_avg",
            "levin_loss_sum",
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
        help="number of gradient steps to be performed in each iteration of the Bootstrap system",
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
        choices=["Levin", "BiLevin"],
        help="name of the search agent",
    )
    parser.add_argument(
        "--batch-size-bootstrap",
        type=int,
        default=32,
        help="number of problems to batch during bootstrap procedure",
    )
    parser.add_argument(
        "--initial-budget",
        type=int,
        default=2**10,
        help="initial budget (nodes expanded) allowed to the bootstrap procedure, or just a budget\
         allowed a non-bootstrap search",
    )
    parser.add_argument(
        "--final-budget",
        type=int,
        default=2**16,
        help="terminate when budget grows at least this large",
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
        "--weight-uniform",
        type=float,
        default="0.0",
        help="mixture weight with a uniform policy",
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
        "--track-params",
        action="store_true",
        default=False,
        help="track basic metrics with tensorboard",
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


if __name__ == "__main__":
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))
    args = parse_args()

    if args.batch_size_bootstrap < world_size:
        raise ValueError(
            f"batch-size-bootstrap '{args.batch_size_bootstrap}' must be >= world_size {world_size} (nnodes * nproc_per_node)"
        )
    if args.batch_size_bootstrap % world_size != 0:
        raise ValueError(
            f"batch-size-bootstrap '{args.batch_size_bootstrap}' must be a multiple of world_size {world_size} (nnodes * nproc_per_node)"
        )
    local_batch_size = args.batch_size_bootstrap // world_size

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

    num_problems_parsed = len(parsed_problems)
    if num_problems_parsed < world_size:
        raise Exception(
            f"Number of problems '{num_problems_parsed}' must be greater than world size '{world_size}'"
        )

    for p in parsed_problems:
        if p[1].is_goal(p[1].initial_state):
            raise Exception(f"Problem '{p[0]}' initial state is a goal state")

    problems_per_process = num_problems_parsed // world_size
    problemsets = []
    for proc in range(world_size):
        problems_local = parsed_problems[
            proc * problems_per_process : (proc + 1) * problems_per_process
        ]
        problemsets.append(problems_local)

    if args.mode == "test":
        problemsets[0].extend(parsed_problems[problems_per_process * world_size :])

    if world_size > 1:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    exp_name = f"_{args.exp_name}" if args.exp_name else ""
    problemset_params = (
        f"{args.problemset_path.parent.stem}-{args.problemset_path.stem}"
    )
    run_name = f"{problemset_dict['domain_name']}-{problemset_params}_{args.agent}-{args.initial_budget}_{args.seed}_{int(start_time)}{args.exp_name}"

    if rank == 0:
        print(time.ctime(start_time))
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

        print(f"Parsed {num_problems_parsed} problems")
        if len(problemsets[0]) == len(problemsets[1]):
            print(
                f"  Loading {problems_per_process * world_size}, {problems_per_process} into each of {world_size} processes\n"
            )
        else:
            print(
                f"  Loading {num_problems_parsed}, {len(problemsets[0])} into rank 0 process, {problems_per_process} into each of {world_size - 1} remaining processes\n"
            )

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

    args.seed += local_rank
    random.seed(args.seed)
    np.random.seed(args.seed)
    to.manual_seed(args.seed)
    if args.torch_deterministic:
        to.use_deterministic_algorithms(True)
        to.backends.cudnn.benchmark = False  # type:ignore
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    agent: Optional[Agent] = None
    if args.agent == "Levin":
        agent = Levin(
            args.weight_uniform,
        )
    elif args.agent == "BiLevin":
        agent = BiLevin(
            args.weight_uniform,
        )
    assert agent is not None

    if args.agent == "Levin":
        model = ConvNetSingle(in_channels, state_t_width, (2, 2), 32, num_actions)
    elif args.agent == "BiLevin":
        forward_model = ConvNetSingle(
            in_channels, state_t_width, (2, 2), 32, num_actions
        )
        if double_backward:
            backward_model = ConvNetDouble(
                in_channels, state_t_width, (2, 2), 32, num_actions
            )
        else:
            backward_model = ConvNetSingle(
                in_channels, state_t_width, (2, 2), 32, num_actions
            )
        model = forward_model, backward_model
    else:
        raise ValueError("Search agent not recognized")

    if args.model_path.is_file():
        if agent.bidirectional:
            forward_model.load_state_dict(to.load(args.model_path))  # type:ignore
            backward_model_path = Path(
                str(args.model_path).replace("_forward.pt", "_backward.pt")
            )
            backward_model.load_state_dict(backward_model_path)  # type:ignore
            if rank == 0:
                print(f"Loaded model\n  from  {str(args.model_path)}")
                print(f"Loaded model\n  from {str(backward_model_path)}")
        else:
            model.load_state_dict(to.load(args.model_path))  # type:ignore
            if rank == 0:
                print(f"Loaded model\n  from  {str(args.model_path)}")
    else:
        if args.model_path.suffix:
            args.model_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            args.model_path.mkdir(parents=True, exist_ok=True)
        args.model_path = Path(args.model_path) / f"{run_name}_forward.pt"
        if rank == 0:
            print(f"Saving model\n  to {str(args.model_path)}")
        if agent.bidirectional:
            backward_model_path = Path(
                str(args.model_path).replace("_forward.pt", "_backward.pt")
            )
            if rank == 0:
                print(f"Saving model\n  to {backward_model_path}")

    # if agent.bidirectional:
    #     model = to.jit.script(model[0]), to.jit.script(model[1])
    # else:
    #     model = to.jit.script(model)

    if rank == 0:
        print(f"World size: {world_size}, rank {rank}")

    if args.mode == "train":
        loss_fn = getattr(loss_fns, args.loss_fn)
        optimizer_cons = to.optim.Adam
        optimizer_params = {
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        }

        train(
            agent,
            model,
            args.model_path,
            loss_fn,
            optimizer_cons,
            optimizer_params,
            problemsets[rank],
            local_batch_size,
            writer,
            world_size,
            initial_budget=args.initial_budget,
            grad_steps=args.grad_steps,
            shuffle_trajectory=args.shuffle_trajectory,
            track_params=args.track_params,
        )

    elif args.mode == "test":
        raise NotImplementedError

    if rank == 0:
        print(f"Total time: {time.time() - start_time}")
        writer.close()
