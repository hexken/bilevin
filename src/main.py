import argparse
import os
from os import listdir
from os.path import isfile, join
import time

import torch as to
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

from bootstrap import Bootstrap
import random
import pathlib
from domains import SlidingTilePuzzle, Sokoban, WitnessState
from models import ModelWrapper
import models.loss_functions as loss_fns
from search import AStar, BFSLevin, GBFS, PUCT


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
        "--problems-folder",
        type=str,
        help="name of folder with problem instances",
    )
    parser.add_argument(
        "-m",
        "--model-folder",
        type=str,
        help="name of folder to load or save NN model",
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
        "--algorithm",
        type=str,
        choices=["Levin", "LevinStar", "PUCT", "AStar", "GBFS"],
        help="name of the search algorithm (Levin, LevinStar, AStar, GBFS, PUCT)",
    )
    parser.add_argument(
        "-k",
        "--batch-size-expansions",
        type=int,
        default=32,
        help="number of nodes to batch for expansion",
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
        choices=["train", "eval"],
        default="train",
        help="train or test the model from model-folder using instances from problems-folder",
    )
    parser.add_argument(
        "--exp-name", type=str, default="bi-levin", help="the name of this experiment"
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
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="if toggled, cuda will be enabled by default",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        default=False,
        help="if toggled, this experiment will be tracked with Weights and Biases",
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


def solve_problems(initial_states, planner, model, time_limit_seconds):
    """ """
    solutions = {}

    for problem_name, initial_state in initial_states.items():
        initial_state.reset()

        solution_depth, expanded, generated, running_time = planner.search(
            initial_state, problem_name, -1, time.time(), time_limit_seconds, 0, model
        )

        solutions[problem_name] = (solution_depth, expanded, generated, running_time)

    for problem_name, data in solutions.items():
        print(
            "{:s}, {:d}, {:d}, {:d}, {:.2f}".format(
                problem_name, data[0], data[1], data[2], data[3]
            )
        )


def solve_problems2(
    initial_states, planner, model, time_limit_seconds, search_budget=-1
):
    """ """
    solutions = {}

    for problem_name, initial_state in initial_states.items():
        initial_state.reset()
        solutions[problem_name] = (-1, -1, -1, -1)

    start_time = time.time()

    while len(initial_states) > 0:

        for problem_name, initial_state in initial_states.items():
            solution_depth, expanded, generated, running_time = planner.search(
                initial_state,
                problem_name,
                search_budget,
                start_time,
                time_limit_seconds,
                model,
            )

            if solution_depth > 0:
                solutions[problem_name] = (
                    solution_depth,
                    expanded,
                    generated,
                    running_time,
                )
                del initial_states[problem_name]

        if (
            time.time() - start_time > time_limit_seconds
            or len(initial_states) == 0
            or search_budget >= 1000000
        ):
            for problem_name, data in solutions.items():
                print(
                    "{:s}, {:d}, {:d}, {:d}, {:.2f}".format(
                        problem_name, data[0], data[1], data[2], data[3]
                    )
                )
            return

        search_budget *= 2


if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()
    run_name = f"{args.domain}__{args.exp_name}__{args.seed}__{int(start_time)}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    to.manual_seed(args.seed)
    if args.torch_deterministic:
        to.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    device = to.device("cuda" if to.cuda.is_available() and args.cuda else "cpu")

    states = {}
    if args.domain == "SlidingTile":
        in_channels = 25
        problem_files = [
            f
            for f in listdir(args.problems_folder)
            if isfile(join(args.problems_folder, f))
        ]

        j = 1
        for filename in problem_files:
            with open(join(args.problems_folder, filename), "r") as file:
                problems = file.readlines()

                for i in range(len(problems)):
                    problem = SlidingTilePuzzle(problems[i])
                    states["problem_" + str(j)] = problem

                    j += 1

    elif args.domain == "Witness":
        in_channels = 9
        problem_files = [
            f
            for f in listdir(args.problems_folder)
            if isfile(join(args.problems_folder, f))
        ]

        j = 1

        for filename in problem_files:
            if "." in filename:
                continue

            with open(join(args.problems_folder, filename), "r") as file:
                problem = file.readlines()

                i = 0
                while i < len(problem):
                    k = i
                    while k < len(problem) and problem[k] != "\n":
                        k += 1
                    s = WitnessState()
                    s.read_state_from_string(problem[i:k])
                    states["problem_" + str(j)] = s
                    i = k + 1
                    j += 1
    #             s.read_state(join(parameters.problems_folder, filename))
    #             states[filename] = s

    elif args.domain == "Sokoban":
        in_channels = 4
        problem = []
        problem_files = []
        if isfile(args.problems_folder):
            problem_files.append(args.problems_folder)
        else:
            problem_files = [
                join(args.problems_folder, f)
                for f in listdir(args.problems_folder)
                if isfile(join(args.problems_folder, f))
            ]

        problem_id = 0

        for filename in problem_files:
            with open(filename, "r") as file:
                all_problems = file.readlines()

            for line_in_problem in all_problems:
                if ";" in line_in_problem:
                    if len(problem) > 0:
                        problem = Sokoban(problem)
                        states["problem_" + str(problem_id)] = problem

                    problem = []
                    problem_id += 1

                elif "\n" != line_in_problem:
                    problem.append(line_in_problem.split("\n")[0])

            if len(problem) > 0:
                problem = Sokoban(problem)
                states["problem_" + str(problem_id)] = problem
    else:
        raise ValueError("problem domain not recognized")

    print("Loaded ", len(states), " instances")

    if args.algorithm == "Levin":
        bfs_planner = BFSLevin(
            args.use_default_heuristic,
            args.use_learned_heuristic,
            False,
            args.batch_size_expansions,
            args.weight_uniform,
        )
    elif args.algorithm == "LevinStar":
        bfs_planner = BFSLevin(
            args.use_default_heuristic,
            args.use_learned_heuristic,
            True,
            args.batch_size_expansions,
            args.weight_uniform,
        )
    elif args.algorithm == "PUCT":

        bfs_planner = PUCT(
            args.use_default_heuristic,
            args.use_learned_heuristic,
            args.batch_size_expansions,
            1,  # todo old cpucnt param, do something
        )
    elif args.algorithm == "AStar":
        bfs_planner = AStar(
            args.use_default_heuristic,
            args.use_learned_heuristic,
            args.batch_size_expansions,
            args.weight_astar,
        )
    elif args.algorithm == "GBFS":
        bfs_planner = GBFS(
            args.use_default_heuristic,
            args.use_learned_heuristic,
            args.batch_size_expansions,
        )
    else:
        raise ValueError("Search algorithm not recognized")

    nn_model = ModelWrapper()

    nn_model.initialize(
        in_channels,
        args.algorithm,
        two_headed_model=args.use_learned_heuristic,
    )

    # todo hacky way of loading a model, in fact the checkpointing could be overhauled
    if pathlib.Path(args.model_folder + "model_weights.pt").is_file():
        nn_model.load_weights(
            join("trained_models_online", args.model_folder, "model_weights")
        )

    # todo this part only works with levin stuff for now
    if args.mode == "train":
        loss_fn = getattr(loss_fns, args.loss_fn)
        optimizer_cons = to.optim.Adam
        optimizer_params = {
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        }

        bootstrap = Bootstrap(
            states,
            args.model_folder,
            loss_fn,
            optimizer_cons,
            optimizer_params,
            initial_budget=args.initial_budget,
            grad_steps=args.grad_steps,
            writer=writer,
        )
        bootstrap.solve_uniform_online(bfs_planner, nn_model)

    elif args.mode == "eval":
        # todo not yet implemented, needs to handle the time_limit and budget args
        solve_problems(
            states,
            bfs_planner,
            nn_model,
            args.time_limit,
        )

    print(f"Total time: {time.time() - start_time}")
