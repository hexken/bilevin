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
from pathlib import Path
import socket


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--min-difficulty-solve-ratio",
        type=float,
        default=0,
        help="advance curriculum when either reached epochs or this ratio of problems solved. 0 to ignore",
    )
    parser.add_argument(
        "--samples-per-difficulty",
        type=int,
        default=None,
        help="advance curriculum when either reached epochs or this ratio of problems solved. 0 to ignore",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="shuffle problems",
    )
    parser.add_argument(
        "--no-backward-goal",
        action="store_true",
        default=False,
        help="do not use backward goal state for policy input",
    )
    parser.add_argument(
        "--n-subgoals",
        type=int,
        default=10,
        help="use a maximum of this many subgoals per trajectory",
    )
    parser.add_argument(
        "--runsdir-path",
        default="runs",
        type=lambda p: Path(p).absolute(),
        help="path of directory to save run results to",
    )
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
        help="suffix of model to load, i.e. model_[suffix].pt",
    )
    parser.add_argument(
        "--loss-fn",
        type=str,
        default="cross_entropy_loss",
        choices=[
            "loop_levin_loss",
            "loop_cross_entropy_loss",
            "merge_cross_entropy_loss",
            "loop_levin_loss_real",
        ],
        help="loss function",
    )
    parser.add_argument(
        "--forward-hidden-layers",
        action="store",
        nargs="+",
        default=[128],
        type=int,
        help="hidden layer sizes of forward policy",
    )
    parser.add_argument(
        "--backward-hidden-layers",
        action="store",
        nargs="+",
        default=[256, 192, 128],
        type=int,
        help="hidden layer sizes of backward policy",
    )
    # parser.add_argument(
    #     "--kernel-dims",
    #     action="store",
    #     nargs="+",
    #     default=[2, 2, 2],
    #     type=int,
    #     help="hidden layer sizes of backward policy",
    # )
    parser.add_argument(
        "--no-jit",
        action="store_true",
        default=False,
        help="do not use torch jit",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.001,
        help="l2 regularization weight",
    )
    parser.add_argument(
        "--share-feature-net",
        action="store_true",
        default=False,
        help="use the same feature netword for forward and backward policies",
    )
    parser.add_argument(
        "--feature-net-lr",
        type=float,
        default=0.001,
        help="feature net learning rate",
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
        "--min-curriculum-epochs",
        type=int,
        default=1,
        help="minimum number of epochs per curriculum difficulty",
    )
    parser.add_argument(
        "--max-curriculum-epochs",
        type=int,
        default=1,
        help="number of curriculum epochs to train for",
    )
    parser.add_argument(
        "--permutation-focus",
        action="store_true",
        default=False,
        help="just use the permutation problems once the bootstrap/curriculum is done",
    )
    parser.add_argument(
        "--permutation-epochs",
        type=int,
        default=1,
        help="number of permutation epochs to train for",
    )
    parser.add_argument(
        "--epoch-reduce-lr",
        type=int,
        default=9999999,
        help="reduce learning rate by a factor of 10 after this many epochs",
    )
    parser.add_argument(
        "--epoch-reduce-grad-steps",
        type=int,
        default=9999999,
        help="reduce number of grad steps by a factor of 2 after this many epochs",
    )
    parser.add_argument(
        "--epoch-begin-validate",
        type=int,
        default=1,
        help="reduce learning rate by a factor of 10 after this many epochs",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=1,
        help="validate every this many epochs",
    )
    parser.add_argument(
        "-a",
        "--agent",
        type=str,
        choices=["Levin", "BiLevin"],
        help="name of the search agent",
    )
    parser.add_argument(
        "--cost-fn",
        type=str,
        default="levin_cost",
        choices=[
            "levin_cost",
        ],
        help="loss function",
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
        "--expansion-budget",
        type=int,
        default=2**10,
        help="initial node expansion budget to solve a problem",
    )
    parser.add_argument(
        "--increase-budget",
        action="store_true",
        default=False,
        help="during testing (not validation), double the budget for each unsolved problem",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=300,
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
