from abc import abstractmethod
from copy import deepcopy
from math import ceil
from pathlib import Path
import pickle as pkl
from queue import Empty
from typing import Optional

import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Queue

from domains.domain import Domain


class Problem:
    def __init__(self, id: int, domain: Domain):
        self.id: int = id
        self.domain: Domain = domain

    def __hash__(self):
        return self.id.__hash__()

    def __eq__(self, other):
        return self.id == other.id


class Loader:
    def __init__(
        self,
        problems: list[Problem],
        batch_size: int | None = None,
        seed: int = 1,
    ):
        if batch_size is None:
            self.batch_size = len(problems)
        else:
            self.batch_size = batch_size
        self.problems = problems
        self.rng = np.random.default_rng(seed)
        self.n_problems = len(self.problems)

    @abstractmethod
    def reset_indices(self, shuffle: bool = False):
        pass

    @abstractmethod
    def next_batch(self):
        pass

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def load_state_dict(self, state: dict):
        pass

    @abstractmethod
    def state_dict(self) -> dict:
        pass

    @classmethod
    @abstractmethod
    def from_path(cls, args, path: Path):
        with path.open("rb") as f:
            pset_dict = pkl.load(f)
        problems = pset_dict["problems"][0]
        indexer = mp.Value("I", 0)
        indices = mp.Array("I", len(problems))
        loader = ArrayLoader(
            problems,
            indices,
            indexer,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        return loader, pset_dict

    def __len__(self):
        return self.n_problems


class QueueLoader(Loader):
    def __init__(
        self,
        problems: list[Problem],
        queue: Queue,
        batch_size: int | None = None,
        seed: int = 1,
    ):
        super().__init__(problems, batch_size, seed)
        self.queue = queue

    @classmethod
    def from_path(cls, args, path: Path):
        with path.open("rb") as f:
            pset_dict = pkl.load(f)
        problems = pset_dict["problems"][0]
        queue = Queue(args.batch_size)
        loader = QueueLoader(
            problems,
            queue,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        return loader, pset_dict

    def get(self):
        try:
            return deepcopy(self.problems[self.queue.get_nowait()])
        except Empty:
            return None

    def reset_indices(self, shuffle: bool = False):
        if shuffle:
            self.indices = self.rng.permutation(self.n_problems)
        else:
            self.indices = np.arange(self.n_problems)

    def next_batch(self, batch: int):
        for i in self.indices[self.batch_size * (batch - 1) : self.batch_size * batch]:
            self.queue.put_nowait(i)

    def load_state_dict(self, state: dict):
        self.indices = state["indices"]
        self.rng.bit_generator.state = state["rng_state"]

    def state_dict(self) -> dict:
        return {
            "indices": self.indices,
            "rng_state": self.rng.bit_generator.state,
        }

    def __len__(self):
        return self.n_problems


class ArrayLoader(Loader):
    def __init__(
        self,
        problems: list[Problem],
        shared_inices,
        shared_indexer,
        batch_size: int | None = None,
        seed: int = 1,
    ):
        super().__init__(problems, batch_size, seed)
        self.s_indices = shared_inices
        self.s_idx = shared_indexer

    def reset_indices(self, shuffle: bool = False):
        if shuffle:
            indices = self.rng.permutation(self.n_problems)
        else:
            indices = np.arange(self.n_problems, dtype=int)

        sentinel_indices = np.arange(
            self.batch_size, self.n_problems + 1, self.batch_size, dtype=int
        )
        if self.n_problems % self.batch_size != 0:
            sentinel_indices = np.append(sentinel_indices, -1)

        mod_indices = np.insert(indices, sentinel_indices, -1)
        del indices, sentinel_indices
        with self.s_indices.get_lock():
            self.s_indices[:] = mod_indices[:]
        del mod_indices

        with self.s_idx.get_lock():
            self.s_idx.value = -1

    def next_batch(self):
        with self.s_idx.get_lock():
            self.s_idx.value += 1

    def get(self):
        with self.s_idx.get_lock():
            i = self.s_indices[self.s_idx.value]
            if i == -1:
                return None
            else:
                self.s_idx.value += 1
        return deepcopy(self.problems[i])

    def load_state_dict(self, state: dict):
        with self.s_idx.get_lock():
            self.s_indices[:] = state["indices"][:]
        with self.s_indices.get_lock():
            self.s_idx.value = state["indexer"]
        self.batch_size = state["batch_size"]
        self.rng.bit_generator.state = state["rng_state"]

    def state_dict(self) -> dict:
        with self.s_idx.get_lock():
            with self.s_indices.get_lock():
                return {
                    "indices": self.s_indices[:],
                    "indexer": self.s_idx.value,
                    "rng_state": self.rng.bit_generator.state,
                }

    @classmethod
    def from_path(cls, args, path: Path):
        with path.open("rb") as f:
            pset_dict = pkl.load(f)
        problems = pset_dict["problems"][0]
        indexer = mp.Value("i", 0)
        n_sentinels = len(problems) // args.batch_size
        if len(problems) % args.batch_size != 0:
            n_sentinels += 1
        indices = mp.Array("i", len(problems) + n_sentinels)
        loader = ArrayLoader(
            problems,
            indices,
            indexer,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        return loader, pset_dict

    def __len__(self):
        return self.n_problems
