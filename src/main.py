import argparse
from os import listdir
from os.path import isfile, join
import time

import torch as to

from bootstrap import Bootstrap
from domains.sliding_tile_puzzle import SlidingTilePuzzle
from domains.sokoban import Sokoban
from domains.witness import WitnessState
import models.loss_functions as loss_fns
from models.model_wrapper import ModelWrapper
from search.a_star import AStar
from search.bfs_levin import BFSLevin
from search.gbfs import GBFS
from search.puct import PUCT


def search_time_limit(initial_states, planner, model, time_limit_seconds):
    """
    This function runs (best-first) Levin tree search with a learned policy on a set of problems.
    The search will be bounded by a time limit. The number of nodes expanded and generated will be
    reported, independently if the planner solved the problem or not. If the planner solves the
    problem, then the procedure also reports solution depth.
    """
    solutions = {}

    # todo: why do prefill the solution dict?
    for puzzle_name, initial_state in initial_states.items():
        initial_state.reset()
        solutions[puzzle_name] = (-1, -1, -1, -1)

        solution_depth, expanded, generated, running_time = planner.search(
            initial_state, puzzle_name, -1, time.time(), time_limit_seconds, 0, model
        )

        solutions[puzzle_name] = (solution_depth, expanded, generated, running_time)

    for puzzle_name, data in solutions.items():
        print(
            "{:s}, {:d}, {:d}, {:d}, {:.2f}".format(
                puzzle_name, data[0], data[1], data[2], data[3]
            )
        )


def search(initial_states, planner, model, time_limit_seconds, search_budget=-1):
    """
    This function runs (best-first) Levin tree search with a learned policy on a set of problems
    """
    slack_time = 600

    solutions = {}

    for puzzle_name, initial_state in initial_states.items():
        initial_state.reset()
        solutions[puzzle_name] = (-1, -1, -1, -1)

    start_time = time.time()

    while len(initial_states) > 0:

        #         args = [(state, name, nn_model, search_budget, start_time, time_limit_seconds, slack_time) for name, state in states.items()]
        #         solution_depth, expanded, generated, running_time, puzzle_name = planner.search(args[0])

        for puzzle_name, initial_state in initial_states.items():
            solution_depth, expanded, generated, running_time = planner.search(
                initial_state,
                puzzle_name,
                search_budget,
                start_time,
                time_limit_seconds,
                slack_time,
                model,
            )

            if solution_depth > 0:
                solutions[puzzle_name] = (
                    solution_depth,
                    expanded,
                    generated,
                    running_time,
                )
                del initial_states[puzzle_name]

        partial_time = time.time()

        if (
            partial_time - start_time + slack_time > time_limit_seconds
            or len(initial_states) == 0
            or search_budget >= 1000000
        ):
            for puzzle_name, data in solutions.items():
                print(
                    "{:s}, {:d}, {:d}, {:d}, {:.2f}".format(
                        puzzle_name, data[0], data[1], data[2], data[3]
                    )
                )
            return

        search_budget *= 2


def main():
    """
    It is possible to use this system to either train a new neural network model through the bootstrap system and
    Levin tree search (LTS) algorithm, or to use a trained neural network with LTS.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-l",
        action="store",
        type=str,
        dest="loss_function",
        default="CrossEntropyLoss",
        help="Loss function",
    )

    parser.add_argument(
        "-lr",
        action="store",
        type=float,
        dest="lr",
        default=0.0001,
        help="Optimizer learning rate",
    )

    parser.add_argument(
        "--weight-decay",
        action="store",
        type=float,
        dest="weight_decay",
        default=0.0,
        help="L2 regularization penalty",
    )

    parser.add_argument(
        "-p",
        action="store",
        type=str,
        dest="problems_folder",
        help="Folder with problem instances",
    )

    parser.add_argument(
        "-m",
        action="store",
        type=str,
        dest="model_name",
        help="Name of the folder of the neural model",
    )

    parser.add_argument(
        "-a",
        type=str,
        action="store",
        dest="search_algorithm",
        help="Name of the search algorithm (Levin, LevinStar, AStar, GBFS, PUCT)",
    )

    parser.add_argument(
        "-d",
        type=str,
        action="store",
        dest="problem_domain",
        help="Problem domain (Witness or SlidingTile)",
    )

    parser.add_argument(
        "-b",
        type=int,
        action="store",
        dest="search_budget",
        default=1000,
        help="The initial budget (nodes expanded) allowed to the bootstrap procedure",
    )

    parser.add_argument(
        "-g",
        type=int,
        action="store",
        dest="gradient_steps",
        default=10,
        help="Number of gradient steps to be performed in each iteration of the Bootstrap system",
    )

    parser.add_argument(
        "-time",
        type=int,
        action="store",
        dest="time_limit",
        default="43200",
        help="Time limit in seconds for search",
    )

    parser.add_argument(
        "-mix",
        type=float,
        action="store",
        dest="mix_epsilon",
        default="0.0",
        help="Mixture with a uniform policy",
    )

    parser.add_argument(
        "-w",
        type=float,
        action="store",
        dest="weight_astar",
        default="1.0",
        help="Weight to be used with WA*.",
    )

    parser.add_argument(
        "--default-heuristic",
        action="store_true",
        default=False,
        dest="use_heuristic",
        help="Use the default heuristic as input",
    )

    parser.add_argument(
        "--learned-heuristic",
        action="store_true",
        default=False,
        dest="use_learned_heuristic",
        help="Use/learn a heuristic",
    )

    parser.add_argument(
        "--blind-search",
        action="store_true",
        default=False,
        dest="blind_search",
        help="Perform blind search",
    )

    parser.add_argument(
        "--learn",
        action="store_true",
        default=False,
        dest="learning_mode",
        help="Train as neural model out of the instances from the problem folder",
    )

    parser.add_argument(
        "--fixed-time",
        action="store_true",
        default=False,
        dest="fixed_time",
        help="Run the planner for a fixed amount of time (specified by time_limit) for each problem instance",
    )

    parser.add_argument(
        "-number-test-instances",
        type=int,
        action="store",
        dest="number_test_instances",
        default="0",
        help="Maximum number of test instances (value of zero will use all instances in the test file).",
    )

    parameters = parser.parse_args()

    states = {}

    if parameters.problem_domain == "SlidingTile":
        in_channels = 25
        puzzle_files = [
            f
            for f in listdir(parameters.problems_folder)
            if isfile(join(parameters.problems_folder, f))
        ]

        j = 1
        for filename in puzzle_files:
            with open(join(parameters.problems_folder, filename), "r") as file:
                problems = file.readlines()

                for i in range(len(problems)):
                    puzzle = SlidingTilePuzzle(problems[i])
                    states["puzzle_" + str(j)] = puzzle

                    j += 1

    elif parameters.problem_domain == "Witness":
        in_channels = 9
        puzzle_files = [
            f
            for f in listdir(parameters.problems_folder)
            if isfile(join(parameters.problems_folder, f))
        ]

        j = 1

        for filename in puzzle_files:
            if "." in filename:
                continue

            with open(join(parameters.problems_folder, filename), "r") as file:
                puzzle = file.readlines()

                i = 0
                while i < len(puzzle):
                    k = i
                    while k < len(puzzle) and puzzle[k] != "\n":
                        k += 1
                    s = WitnessState()
                    s.read_state_from_string(puzzle[i:k])
                    states["puzzle_" + str(j)] = s
                    i = k + 1
                    j += 1
    #             s.read_state(join(parameters.problems_folder, filename))
    #             states[filename] = s

    elif parameters.problem_domain == "Sokoban":
        in_channels = 4
        problem = []
        puzzle_files = []
        if isfile(parameters.problems_folder):
            puzzle_files.append(parameters.problems_folder)
        else:
            puzzle_files = [
                join(parameters.problems_folder, f)
                for f in listdir(parameters.problems_folder)
                if isfile(join(parameters.problems_folder, f))
            ]

        problem_id = 0

        for filename in puzzle_files:
            with open(filename, "r") as file:
                all_problems = file.readlines()

            for line_in_problem in all_problems:
                if ";" in line_in_problem:
                    if len(problem) > 0:
                        puzzle = Sokoban(problem)
                        states["puzzle_" + str(problem_id)] = puzzle

                    problem = []
                    #                 problem_id = line_in_problem.split(' ')[1].split('\n')[0]
                    problem_id += 1

                elif "\n" != line_in_problem:
                    problem.append(line_in_problem.split("\n")[0])

            if len(problem) > 0:
                puzzle = Sokoban(problem)
                states["puzzle_" + str(problem_id)] = puzzle
    else:
        raise ValueError("Problem domain not recognized")

    if parameters.number_test_instances != 0:
        states_capped = {}
        counter = 0

        for name, puzzle in states.items():
            states_capped[name] = puzzle
            counter += 1

            if counter == parameters.number_test_instances:
                break

        states = states_capped

    print("Loaded ", len(states), " instances")
    #     input_size = s.get_image_representation().shape

    k_expansions = 32

    start = time.time()

    nn_model = ModelWrapper()
    bootstrap = None

    if parameters.learning_mode:

        if parameters.loss_function == "LevinLoss":
            parameters.loss_function = "traj_levin_loss"

        loss_fn = getattr(loss_fns, parameters.loss_function)
        optimizer_cons = to.optim.Adam
        optimizer_params = {
            "lr": parameters.lr,
            "weight_decay": parameters.weight_decay,
        }

        bootstrap = Bootstrap(
            states,
            parameters.model_name,
            loss_fn,
            optimizer_cons,
            optimizer_params,
            initial_budget=parameters.search_budget,
            gradient_steps=parameters.gradient_steps,
        )

    if (
        parameters.search_algorithm == "Levin"
        or parameters.search_algorithm == "LevinStar"
    ):

        if parameters.search_algorithm == "Levin":
            bfs_planner = BFSLevin(
                parameters.use_heuristic,
                parameters.use_learned_heuristic,
                False,
                k_expansions,
                parameters.mix_epsilon,
            )
        else:
            bfs_planner = BFSLevin(
                parameters.use_heuristic,
                parameters.use_learned_heuristic,
                True,
                k_expansions,
                parameters.mix_epsilon,
            )

        if parameters.use_learned_heuristic:
            nn_model.initialize(
                in_channels,
                parameters.search_algorithm,
                two_headed_model=True,
            )
        else:
            nn_model.initialize(
                in_channels,
                parameters.search_algorithm,
                two_headed_model=False,
            )

        if parameters.learning_mode:
            bootstrap.solve_uniform_online(bfs_planner, nn_model)
        elif parameters.blind_search:
            search(
                states,
                bfs_planner,
                nn_model,
                parameters.time_limit,
                parameters.search_budget,
            )
        elif parameters.fixed_time:
            nn_model.load_weights(
                join("trained_models_online", parameters.model_name, "model_weights")
            )
            search_time_limit(states, bfs_planner, nn_model, parameters.time_limit)
        else:
            nn_model.load_weights(
                join("trained_models_online", parameters.model_name, "model_weights")
            )
            search(
                states,
                bfs_planner,
                nn_model,
                parameters.time_limit,
                parameters.search_budget,
            )
    elif parameters.search_algorithm == "PUCT":

        bfs_planner = PUCT(
            parameters.use_heuristic,
            parameters.use_learned_heuristic,
            k_expansions,
            1,  # todo old cpucnt param, do something
        )

        if parameters.use_learned_heuristic:
            nn_model.initialize(
                in_channels,
                parameters.search_algorithm,
                two_headed_model=True,
            )
        else:
            nn_model.initialize(
                in_channels,
                parameters.search_algorithm,
                two_headed_model=False,
            )

        if parameters.learning_mode:
            bootstrap.solve_uniform_online(bfs_planner, nn_model)
        elif parameters.blind_search:
            search(
                states,
                bfs_planner,
                nn_model,
                parameters.time_limit,
                parameters.search_budget,
            )
        elif parameters.fixed_time:
            nn_model.load_weights(
                join("trained_models_online", parameters.model_name, "model_weights")
            )
            search_time_limit(states, bfs_planner, nn_model, parameters.time_limit)
        else:
            nn_model.load_weights(
                join("trained_models_online", parameters.model_name, "model_weights")
            )
            search(
                states,
                bfs_planner,
                nn_model,
                parameters.time_limit,
                parameters.search_budget,
            )

    elif parameters.search_algorithm == "AStar":
        bfs_planner = AStar(
            parameters.use_heuristic,
            parameters.use_learned_heuristic,
            k_expansions,
            float(parameters.weight_astar),
        )

        if parameters.learning_mode and parameters.use_learned_heuristic:
            nn_model.initialize(in_channels, parameters.search_algorithm)
            bootstrap.solve_uniform_online(bfs_planner, nn_model)
        elif parameters.fixed_time and parameters.use_learned_heuristic:
            nn_model.initialize(in_channels, parameters.search_algorithm)
            nn_model.load_weights(
                join("trained_models_online", parameters.model_name, "model_weights")
            )
            search_time_limit(states, bfs_planner, nn_model, parameters.time_limit)
        elif parameters.use_learned_heuristic:
            nn_model.initialize(in_channels, parameters.search_algorithm)
            nn_model.load_weights(
                join("trained_models_online", parameters.model_name, "model_weights")
            )
            search(
                states,
                bfs_planner,
                nn_model,
                parameters.time_limit,
                parameters.search_budget,
            )
        else:
            search(
                states,
                bfs_planner,
                nn_model,
                parameters.time_limit,
                parameters.search_budget,
            )

    elif parameters.search_algorithm == "GBFS":
        bfs_planner = GBFS(
            parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions
        )

        if parameters.learning_mode:
            nn_model.initialize(in_channels, parameters.search_algorithm)
            bootstrap.solve_uniform_online(bfs_planner, nn_model)
        elif parameters.fixed_time and parameters.use_learned_heuristic:
            nn_model.initialize(in_channels, parameters.search_algorithm)
            nn_model.load_weights(
                join("trained_models_online", parameters.model_name, "model_weights")
            )
            search_time_limit(states, bfs_planner, nn_model, parameters.time_limit)
        elif parameters.use_learned_heuristic:
            nn_model.initialize(in_channels, parameters.search_algorithm)
            nn_model.load_weights(
                join("trained_models_online", parameters.model_name, "model_weights")
            )
            search(
                states,
                bfs_planner,
                nn_model,
                parameters.time_limit,
                parameters.search_budget,
            )
        else:
            search(
                states,
                bfs_planner,
                nn_model,
                parameters.time_limit,
                parameters.search_budget,
            )

    print("Total time: ", time.time() - start)


if __name__ == "__main__":
    main()
