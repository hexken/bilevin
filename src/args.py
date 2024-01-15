import argparse
from pathlib import Path
import socket


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weight-astar",
        type=float,
        default=1,
        help="weight to use for weighted A*",
    )
    parser.add_argument(
        "--n-final-stage-epochs",
        type=int,
        default=1,
        help="number of epochs to train on final curriculum stage",
    )
    parser.add_argument(
        "--min-solve-ratio-stage",
        type=float,
        default=0,
        help="advance curriculum if the last n-tail problems has at least this solve ratio and at least min-problems-per-stage problems have been attempted",
    )
    parser.add_argument(
        "--min-solve-ratio-exp",
        type=float,
        default=0,
        help="increase budget during training if the last n-tail stage problems has below this solve ratio and at least min-problems-per-stage problems have been attempted. Budget resets with each curriculum stage.",
    )
    parser.add_argument(
        "--n-tail",
        type=int,
        default=0,
        help="compute solve ratios based on last n problems. 0 to use all stage problems seen till so far.",
    )
    parser.add_argument(
        "--min-problems-per-stage",
        type=int,
        default=-1,
        help="minimum number of problems to attempt for each curriculum stage. Set to 0 for no minimum. Set to -1 to use the number of problems in the stage.",
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
        "--problems-path",
        type=lambda p: Path(p).absolute(),
        help="path of file with training or test problem instances",
    )
    parser.add_argument(
        "--trainstate-path",
        type=lambda p: Path(p).absolute(),
        help="Continue training from this prior run",
    )
    parser.add_argument(
        "--valid-path",
        type=lambda p: Path(p).absolute(),
        help="path of file with valid problem instances",
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
        help="suffix of model to load, i.e. model_[suffix].pt",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["SGD", "Adam"],
        help="torch optimizer to use",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=-1,
        help="max norm of gradients, -1 to disable",
    )
    parser.add_argument(
        "--loss-fn",
        type=str,
        default="levin_loss",
        choices=[
            "levin_sum_mse_loss",
            "levin_avg_mse_loss",
            "cross_entropy_loss",
            "cross_entropy_mid_loss",
            "cross_entropy_mse_loss",
            "mse_loss",
        ],
        help="loss function",
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
        default=[256, 192, 128],
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
        default=[256, 192, 128],
        type=int,
        help="hidden layer sizes of backward heruistic",
    )
    parser.add_argument(
        "--kernel-size",
        action="store",
        nargs=2,
        type=int,
        required=True,
        help="depth x height/width of convolution kernel",
    )
    parser.add_argument(
        "--num-kernels",
        action="store",
        default=32,
        type=int,
        help="number of convolution kernels",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.001,
        help="add L2 regularization to loss",
    )
    parser.add_argument(
        "--conditional-backward",
        action="store_true",
        default=False,
        help="pass the problems initial (forward) state to the backward policy/heuristic in addition to a current (backward) state",
    )
    parser.add_argument(
        "--no-feature-net",
        action="store_true",
        default=False,
        help="do not use a feature net to extract features from states",
    )
    parser.add_argument(
        "--share-feature-net",
        action="store_true",
        default=False,
        help="use the same feature netword for forward and backward policies/heuristics. In this case forward-feature-net-lr is used",
    )
    parser.add_argument(
        "--keep-all-checkpoints",
        action="store_true",
        default=False,
        help="save all checkpoints instead of just the most recent",
    )
    parser.add_argument(
        "--forward-feature-net-lr",
        type=float,
        default=0.001,
        help="forward feature net learning rate, if not sharing feature net",
    )
    parser.add_argument(
        "--backward-feature-net-lr",
        type=float,
        default=0.001,
        help="backward feature net learning rate, if not sharing feature net",
    )
    parser.add_argument(
        "--forward-policy-lr",
        type=float,
        default=0.001,
        help="forward policy learning rate",
    )
    parser.add_argument(
        "--backward-policy-lr",
        type=float,
        default=0.001,
        help="backward policu learning rate",
    )
    parser.add_argument(
        "--forward-heuristic-lr",
        type=float,
        default=0.001,
        help="forward heuristic learning rate",
    )
    parser.add_argument(
        "--backward-heuristic-lr",
        type=float,
        default=0.001,
        help="backward heuristic learning rate",
    )
    parser.add_argument(
        "-g",
        "--grad-steps",
        type=int,
        default=10,
        help="number of gradient steps to be performed in each opt pass",
    )
    # parser.add_argument(
    #     "--epoch-reduce-lr",
    #     type=int,
    #     default=9999999,
    #     help="reduce learning rate by a factor of 10 after this many epochs",
    # )
    # parser.add_argument(
    #     "--epoch-reduce-grad-steps",
    #     type=int,
    #     default=9999999,
    #     help="reduce number of grad steps by a factor of 2 after this many epochs",
    # )
    parser.add_argument(
        "--batch-begin-validate",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=100,
        help="validate every this many batches",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="validate every this many batches",
    )
    parser.add_argument(
        "-a",
        "--agent",
        type=str,
        choices=[
            "Levin",
            "BiLevinBFS",
            "BiLevinAlt",
            "AStar",
            "BiAStarBFS",
            "BiAStarAlt",
            "PHS",
            "BiPHSBFS",
            "BiPHSAlt",
        ],
        help="name of the search agent to use",
    )
    # parser.add_argument(
    #     "--cost-fn",
    #     type=str,
    #     default="levin_cost",
    #     choices=[
    #         "levin_cost",
    #     ],
    #     help="cost function for best-first search",
    # )
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
        "--max-expansion-budget",
        type=int,
        default=100000000,
        help="initial node expansion budget to solve a problem during training",
    )
    parser.add_argument(
        "--train-expansion-budget",
        type=int,
        help="initial node expansion budget to solve a problem during training",
    )
    parser.add_argument(
        "--test-expansion-budget",
        type=int,
        help="initial node expansion budget to solve a problem during testing/validation",
    )
    # parser.add_argument(
    #     "--increase-budget",
    #     action="store_true",
    #     default=False,
    #     help="during training double the budget every min-problems-per-stage when the solve ratio is below min-solve-ratio-exp, and reset to expansion-budget at beginning of each stage",
    # )
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
