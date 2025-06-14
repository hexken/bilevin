import argparse
from pathlib import Path
import socket


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    from https://github.com/python/cpython/blob/v3.11.2/Lib/distutils/util.py#L308
    """
    lval = val.lower()
    if lval in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif lval in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"invalid truth value {val}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shuffle",
        const=True,
        nargs="?",
        type=strtobool,
        default=False,
        help="shuffle training problems each epoch",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=32,
        help="number of nodes to evaluate in a single batch",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="number of problems in attempted before a model update",
    )
    parser.add_argument(
        "--weight-mse-loss",
        type=float,
        default=1.0,
        help="weight to use for mse loss when agent has a policy and heurisic",
    )
    parser.add_argument(
        "--weight-astar",
        type=float,
        default=1.0,
        help="weight to use for weighted A*",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=10, help="number of epochs to train for"
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        type=lambda p: Path(p).absolute(),
        help="path of checkpoint file",
    )
    parser.add_argument(
        "--runsdir-path",
        default="runs",
        type=lambda p: Path(p).absolute(),
        help="path of directory to save run results to",
    )
    parser.add_argument(
        "--train-path",
        type=lambda p: Path(p).absolute(),
        help="path of file with training problem instances",
    )
    parser.add_argument(
        "--valid-path",
        type=lambda p: Path(p).absolute(),
        help="path of file with valid problem instances",
    )
    parser.add_argument(
        "--test-path",
        type=lambda p: Path(p).absolute(),
        help="path of file with test problem instances",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=lambda p: Path(p).absolute(),
        default=None,
        help="path of directory to load previously saved model(s) from",
    )
    parser.add_argument(
        "--test-suffix",
        type=str,
        default="best_expanded",
        choices=["best_expanded", "final"],
        help="type of model to use for test",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["SGD", "Adam"],
        help="torch optimizer to use",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.001,
        help="add L2 regularization to loss",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=-1,
        help="max norm of gradients, -1 for no constraint",
    )
    parser.add_argument(
        "--loss-fn",
        type=str,
        default="default",
        choices=[
            "default",
            "nll_sum",
            "nll_avg",
            "levin_sum",
            "levin_avg",
            "mse",
        ],
        help="loss function",
    )
    parser.add_argument(
        "--forward-feature-layers",
        action="store",
        nargs="+",
        default=[128, 128],
        type=int,
        help="hidden layer sizes of forward policy",
    )
    parser.add_argument(
        "--backward-feature-layers",
        action="store",
        nargs="+",
        default=[128, 128],
        type=int,
        help="hidden layer sizes of backward policy",
    )
    parser.add_argument(
        "--forward-policy-layers",
        action="store",
        nargs="+",
        default=[128],
        type=int,
        help="hidden layer sizes of forward policy",
    )
    parser.add_argument(
        "--backward-policy-layers",
        action="store",
        nargs="+",
        default=[256, 198, 128],
        type=int,
        help="hidden layer sizes of backward policy",
    )
    parser.add_argument(
        "--forward-heuristic-layers",
        action="store",
        nargs="+",
        default=[128],
        type=int,
        help="hidden layer sizes of forward heruistic",
    )
    parser.add_argument(
        "--backward-heuristic-layers",
        action="store",
        nargs="+",
        default=[256, 198, 128],
        type=int,
        help="hidden layer sizes of backward heruistic",
    )
    parser.add_argument(
        "--kernel-size",
        action="store",
        nargs=2,
        type=int,
        default=[1, 2],
        help="depth x height/width of convolution kernel",
    )
    parser.add_argument(
        "--n-kernels",
        action="store",
        default=32,
        type=int,
        help="number of convolution kernels",
    )
    parser.add_argument(
        "--mask-invalid-actions",
        const=True,
        nargs="?",
        type=strtobool,
        default=False,
        help="mask invalid actions in policy output (assign logits -1e9)",
    )
    parser.add_argument(
        "--feature-net-type",
        type=str,
        default="conv",
        choices=[
            "conv",
            "linear",
        ],
        help="type of feature net to use",
    )
    parser.add_argument(
        "--n-embed-dim",
        action="store",
        default=128,
        type=int,
        help="size of linear embedding",
    )
    parser.add_argument(
        "--no-feature-net",
        const=True,
        nargs="?",
        type=strtobool,
        default=False,
        help="do not use a feature net to extract features from states",
    )
    parser.add_argument(
        "--share-feature-net",
        const=True,
        nargs="?",
        type=strtobool,
        default=False,
        help="use the same feature netword for forward and backward policies/heuristics. In this case forward-feature-net-lr is used",
    )
    parser.add_argument(
        "--master-lr",
        type=float,
        default=-1,
        help="if > 0, use this lr for all networks",
    )
    parser.add_argument(
        "--forward-feature-net-lr",
        type=float,
        default=0.0001,
        help="forward feature net learning rate, if not sharing feature net",
    )
    parser.add_argument(
        "--backward-feature-net-lr",
        type=float,
        default=0.0001,
        help="backward feature net learning rate, if not sharing feature net",
    )
    parser.add_argument(
        "--forward-policy-lr",
        type=float,
        default=0.0001,
        help="forward policy learning rate",
    )
    parser.add_argument(
        "--backward-policy-lr",
        type=float,
        default=0.0001,
        help="backward policu learning rate",
    )
    parser.add_argument(
        "--forward-heuristic-lr",
        type=float,
        default=0.0001,
        help="forward heuristic learning rate",
    )
    parser.add_argument(
        "--backward-heuristic-lr",
        type=float,
        default=0.0001,
        help="backward heuristic learning rate",
    )
    parser.add_argument(
        "-g",
        "--grad-steps",
        type=int,
        default=10,
        help="number of gradient steps to be performed in each opt pass",
    )
    parser.add_argument(
        "--checkpoint-every-n-batch",
        type=int,
        default=100,
        help="checkpoint every this many batches",
    )
    parser.add_argument(
        "-a",
        "--agent",
        type=str,
        choices=[
            "Levin",
            "BiLevin",
            "BiLevinBFS",
            "BiLevinAlt",
            "AStar",
            "BiAStar",
            "BiAStarBFS",
            "BiAStarAlt",
            "PHS",
            "BiPHS",
            "BiPHSBFS",
            "BiPHSAlt",
        ],
        help="name of the search agent to use",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=4,
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
        default="auto",
        help="port for multiprocessing communication",
    )
    parser.add_argument(
        "--lockfile",
        type=str,
        default="/tmp/port",
        help="path to create lock file for coordinating port setting in single-node multi instance training",
    )
    parser.add_argument(
        "--train-expansion-budget",
        type=int,
        help="initial node expansion budget to solve a problem during training",
    )
    parser.add_argument(
        "--test-expansion-budget",
        type=int,
        default=-1,
        help="initial node expansion budget to solve a problem during testing/validation, -1 to use train expansion budget",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=86400,
        help="time budget to solve a problem",
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
    args = parser.parse_args()
    return args
