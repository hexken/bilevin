import argparse
import os
from pathlib import Path
import random
import time

import numpy as np
import torch as to
import torch.distributed as dist
from torch.utils.tensorboard.writer import SummaryWriter

from domains import SlidingTilePuzzle, Sokoban, WitnessState
from models import (
    ConvNetDouble,
    ConvNetSingle,
    HeuristicConvNetSingle,
    TwoHeadedConvNetSingle,
)
import models.loss_functions as loss_fns
from search import AStar, BiLevin, GBFS, Levin, PUCT
from test import test
from train import train
from utils import rank0_print


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--domain",
        type=str,
        choices=["SlidingTile", "Witness", "Sokoban"],
        help="problem domain",
    )
    parser.add_argument(
        "-p",
        "--problems-path",
        type=lambda p: Path(p).absolute(),
        help="path of directory with problem instances",
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
        default="levin_loss",
        choices=[
            "levin_loss",
            "improved_levin_loss",
            "mse_loss",
            "cross_entropy_loss",
        ],
        help="loss function",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
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
        "-a",
        "--agent",
        type=str,
        choices=["Levin", "BiLevin", "LevinStar", "PUCT", "AStar", "GBFS"],
        help="name of the search agent",
    )
    parser.add_argument(
        "--batch-size-expansions",
        type=int,
        default=32,
        help="number of nodes to batch for expansion",
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
    parser.add_argument(
        "--use-default-heuristic",
        action="store_true",
        help="use the default heuristic",
    )
    parser.add_argument(
        "--use-learned-heuristic",
        action="store_true",
        default=False,
        help="use the learned heuristic",
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
        "--wandb",
        action="store_true",
        default=False,
        help="track with Weights and Biases",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="bi-levin",
        help="the wandb's project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="the entity (team) of wandb's project",
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
    exp_name = f"_{args.exp_name}" if args.exp_name else ""
    run_name = (
        f"{args.domain}_{args.agent}_{args.seed}_{int(start_time)}{args.exp_name}"
    )

    problems = []
    if args.domain == "SlidingTile":
        problems_gathered = []
        for file in args.problems_path.iterdir():
            print()
            problems_gathered.extend(
                [SlidingTilePuzzle(p) for p in file.read_text().splitlines()]
            )

        problems_per_process = len(problems_gathered) // world_size
        for proc in range(world_size):
            problems_local = {
                f"p{i + 1}": p
                for i, p in enumerate(
                    problems_gathered[
                        proc * problems_per_process : (proc + 1) * problems_per_process
                    ],
                    start=proc * problems_per_process,
                )
            }
            problems.append(problems_local)
        in_channels = problems[0]["p1"].getSize()
    else:
        # todo add other domains
        raise ValueError("problem domain not recognized")

    if world_size > 1:
        backend = "nccl" if args.cuda and to.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if rank == 0:
        print(
            f"Loaded {len(problems_gathered)} total problems\n  {problems_per_process} into each of {world_size} processes"
        )
        if args.wandb:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                save_code=True,
            )
            print("wandb initialized")

        print(f"Logging with tensorboard\n  to {run_name}")
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    else:
        writer = SummaryWriter()

    args.seed += local_rank
    random.seed(args.seed)
    np.random.seed(args.seed)
    to.manual_seed(args.seed)
    if args.torch_deterministic:
        to.use_deterministic_agents(True)
        to.backends.cudnn.benchmark = False  # type:ignore
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if len(args.device_ids) > 0:
        assert (
            len(args.device_ids) == world_size
        ), "you must specify the same number of device ids as `--nproc_per_node`"
        device = to.device(
            f"cuda:{args.device_ids[local_rank]}"
            if to.cuda.is_available() and args.cuda
            else "cpu"
        )
    else:
        device_count = to.cuda.device_count()
        if device_count < world_size:
            device = to.device(
                "cuda" if to.cuda.is_available() and args.cuda else "cpu"
            )
        else:
            device = to.device(
                f"cuda:{local_rank}" if to.cuda.is_available() and args.cuda else "cpu"
            )

    if args.agent == "Levin":
        agent = Levin(
            args.use_default_heuristic,
            args.use_learned_heuristic,
            False,
            args.batch_size_expansions,
            args.weight_uniform,
        )
    elif args.agent == "LevinStar":
        agent = Levin(
            args.use_default_heuristic,
            args.use_learned_heuristic,
            True,
            args.batch_size_expansions,
            args.weight_uniform,
        )
    elif args.agent == "BiLevin":
        agent = BiLevin(
            args.use_default_heuristic,
            args.use_learned_heuristic,
            False,
            args.batch_size_expansions,
            args.weight_uniform,
        )
    elif args.agent == "PUCT":

        agent = PUCT(
            args.use_default_heuristic,
            args.use_learned_heuristic,
            args.batch_size_expansions,
            1,  # todo old cpucnt param, do something
        )
    elif args.agent == "AStar":
        agent = AStar(
            args.use_default_heuristic,
            args.use_learned_heuristic,
            args.batch_size_expansions,
            args.weight_astar,
        )
    elif args.agent == "GBFS":
        agent = GBFS(
            args.use_default_heuristic,
            args.use_learned_heuristic,
            args.batch_size_expansions,
        )
    else:
        raise ValueError("Search agent not recognized")

    bidirectional = False
    num_actions = 4
    if (
        args.agent == "Levin"
        or args.agent == "LevinMult"
        or args.agent == "LevinStar"
        or args.agent == "PUCT"
        or args.agent == "BiLevin"
    ):
        if args.agent == "BiLevin":
            forward_model = ConvNetSingle(in_channels, (2, 2), 32, num_actions).to(
                device
            )
            backward_model = ConvNetDouble(in_channels, (2, 2), 32, num_actions).to(
                device
            )
            bidirectional = True
            model = (forward_model, backward_model)
        elif args.use_learned_heuristic:
            model = TwoHeadedConvNetSingle(in_channels, (2, 2), 32, num_actions).to(
                device
            )
        else:
            model = ConvNetSingle(in_channels, (2, 2), 32, num_actions).to(device)
    elif args.agent == "AStar" or args.agent == "GBFS":
        model = HeuristicConvNetSingle(in_channels, (2, 2), 32, num_actions).to(device)
    else:
        raise ValueError("Search agent not recognized")

    if args.model_path.is_file():
        if bidirectional:
            forward_model.load_state_dict(to.load(args.model_path))  # type:ignore
            backward_model_path = Path(
                str(args.model_path).replace("_forward.pt", "_backward.pt")
            )
            backward_model.load_state_dict(backward_model_path)  # type:ignore
            rank0_print(f"Loaded model\n  from  {str(args.model_path)}", rank)
            rank0_print(f"Loaded model\n  from {str(backward_model_path)}", rank)
        else:
            model.load_state_dict(to.load(args.model_path))  # type:ignore
            rank0_print(f"Loaded model\n  from  {str(args.model_path)}", rank)
    else:
        if args.model_path.suffix:
            args.model_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            args.model_path.mkdir(parents=True, exist_ok=True)
        args.model_path = Path(args.model_path) / f"{run_name}_forward.pt"
        rank0_print(f"Saving model\n  to {str(args.model_path)}", rank)
        if bidirectional:
            backward_model_path = Path(
                str(args.model_path).replace("_forward.pt", "_backward.pt")
            )
            rank0_print(f"Saving model\n  to {backward_model_path}", rank)

    if world_size > 1:
        dist.barrier()

    print(f"Rank {rank} using device: {device}")

    if world_size > 1:
        dist.barrier()

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
            batch_size=args.batch_size_bootstrap,
        )

    elif args.mode == "test":
        test(
            problems,
            agent,
            model,
            args.time_limit,
        )

    print(f"Total time: {time.time() - start_time}")
    writer.close()
