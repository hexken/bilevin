from copy import deepcopy
from typing import Optional

import numpy as np
import torch.multiprocessing as mp

from domains.domain import Domain


class Problem:
    def __init__(self, id: int, domain: Domain):
        self.id: int = id
        self.domain: Domain = domain

    def __hash__(self):
        return self.id.__hash__()

    def __eq__(self, other):
        return self.id == other.id


class AsyncProblemLoader:
    def __init__(
        self,
        problems: list[Problem],
        shared_inices,
        shared_indexer,
        batch_size: int | None = None,
        seed: int = 1,
    ):
        if batch_size is None:
            self.batch_size = len(problems)
        else:
            self.batch_size = batch_size
        self.problems = problems
        self.shared_indices = shared_inices
        self.shared_indexer = shared_indexer

        self.rng = np.random.default_rng(seed)
        self.loaded_state = False
        self.n_problems = len(self.problems)

    def init_indexer(self, shuffle: bool = False):
        with self.shared_indices.get_lock():
            with self.shared_indexer.get_lock():
                if shuffle:
                    new_indices = self.rng.permutation(self.n_problems)
                else:
                    new_indices = np.arange(self.n_problems)
                self.shared_indices[:] = new_indices[:]
                self.shared_indexer.value = 0

    def advance_batch(self):
        with self.shared_indices.get_lock():
            with self.shared_indexer.get_lock():
                idx = self.shared_indexer.value
                problem = self.problems[self.shared_indices[idx]]
                self.shared_indexer.value += 1
                return problem

    def get_problem(self):
        with self.shared_indices.get_lock():
            with self.shared_indexer.get_lock():
                idx = self.shared_indexer.value
                if idx == self.n_problems or idx % self.batch_size == 0:
                    return None
                else:
                    problem = self.problems[self.shared_indices[idx]]
                    self.shared_indexer.value += 1
                    return problem

    def load_state_dict(self, state: dict):
        with self.shared_indexer.get_lock():
            with self.shared_indices.get_lock():
                self.shared_indices[:] = state["indices"][:]
                self.shared_indexer.value = state["indexer"]
        self.batch_size = state["batch_size"]
        self.rng = state["rng"]

    def state_dict(self) -> dict:
        with self.shared_indexer.get_lock():
            with self.shared_indices.get_lock():
                return {
                    "indices": self.shared_indices[:],
                    "indexer": self.shared_indexer.value,
                    "batch_size": self.batch_size,
                    "rng": self.rng,
                }

    def __len__(self):
        return self.n_problems
