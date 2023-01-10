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
        default=1024,
        help="initial budget (nodes expanded) allowed to the bootstrap procedure, or just a budget\
         allowed a non-bootstrap search",
    )
    parser.add_argument(
        "--final-budget",
        type=int,
        default=2000000,
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
        "-w",
        "--weight-astar",
        type=float,
        default="1.0",
        help="weight to be used with WA*.",
    )
    # parser.add_argument(
    #     "--use-default-heuristic",
    #     action="store_true",
    #     help="use the default heuristic",
    # )
    # parser.add_argument(
    #     "--use-learned-heuristic",
    #     action="store_true",
    #     default=False,
    #     help="use the learned heuristic",
    # )
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
        "--cuda",
        action="store_true",
        help="enable cuda",
    )
    parser.add_argument(
        "--device-ids",
        nargs="+",
        default=[],
        help="the device ids that subprocess workers will use",
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
    if args.batch_size_bootstrap % world_size != 0:
        raise ValueError(
            "batch-size-bootstrap must be a multiple of world-size (nnodes * nproc_per_node)"
        )
    start_time = time.time()

    problemset_dict = json.load(args.problemset_path.open("r"))

    domain_module = getattr(domains, problemset_dict["domain_module"])
    (all_problems, num_actions, in_channels, state_t_width, double_backward,) = getattr(
        domain_module, "load_problemset"
    )(problemset_dict)

    problems_per_process = 0

    assert len(all_problems) > 0, "No problems were loaded"

    for p in all_problems:
        assert not p[1].is_goal(p[1].initial_state)

    problems = []
    problems_per_process = len(all_problems) // world_size
    for proc in range(world_size):
        problems_local = all_problems[
            proc * problems_per_process : (proc + 1) * problems_per_process
        ]
        problems.append(problems_local)

    if world_size > 1:
        backend = "nccl" if args.cuda and to.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    exp_name = f"_{args.exp_name}" if args.exp_name else ""
    run_name = f"{problemset_dict['domain_name']}_{args.agent}_{args.seed}_{int(start_time)}{args.exp_name}"

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

        print(f"Logging with tensorboard\n  to runs/{run_name}")

        print(
            f"Parsed {len(all_problems)} problems\n"
            f"  Loaded {problems_per_process * world_size}, {problems_per_process} into each of {world_size} processes"
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

    device: to.device = to.device("cpu")
    if args.cuda and not to.cuda.is_available():
        raise ValueError("cuda is not available")
    elif args.cuda and len(args.device_ids) < world_size:
        raise ValueError(
            "you must specify the same number of device ids as `--nproc_per_node`"
        )
    elif args.cuda and len(args.device_ids) >= world_size:
        device = to.device(f"cuda:{local_rank}")

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
        model = ConvNetSingle(in_channels, state_t_width, (2, 2), 32, num_actions).to(
            device
        )
    elif args.agent == "BiLevin":
        forward_model = ConvNetSingle(
            in_channels, state_t_width, (2, 2), 32, num_actions
        ).to(device)
        if double_backward:
            backward_model = ConvNetDouble(
                in_channels, state_t_width, (2, 2), 32, num_actions
            ).to(device)
        else:
            backward_model = ConvNetSingle(
                in_channels, state_t_width, (2, 2), 32, num_actions
            ).to(device)
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

    if rank == 0:
        print(f"World size: {world_size}, rank {rank} using device: {device}")

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
            problems[rank],
            writer,
            world_size,
            initial_budget=args.initial_budget,
            grad_steps=args.grad_steps,
            shuffle_trajectory=args.shuffle_trajectory,
            batch_size=args.batch_size_bootstrap,
            track_params=args.track_params,
        )

    elif args.mode == "test":
        raise NotImplementedError

    if rank == 0:
        print(f"Total time: {time.time() - start_time}")
        writer.close()
