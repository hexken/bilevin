from copy import deepcopy
from pathlib import Path
import pickle as pkl

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


class ArrayLoader:
    def __init__(
        self,
        problems: list[Problem],
        shared_inices,
        shared_indexer,
        batch_size: int,
        full_batches: int,
        seed: int,
    ):
        self.batch_size = batch_size
        self.problems = problems
        self.rng = np.random.default_rng(seed)
        self.n_problems = len(self.problems)
        self.s_indices = shared_inices
        self.s_idx = shared_indexer
        self.full_batches = full_batches
        self.remainder_batch_size = self.n_problems % self.batch_size

    def reset_indices(self, shuffle: bool = False):
        if shuffle:
            indices = self.rng.permutation(self.n_problems)
        else:
            indices = np.arange(self.n_problems, dtype=int)

        sentinel_indices = np.arange(
            self.batch_size, self.n_problems + 1, self.batch_size, dtype=int
        )
        mod_indices = np.insert(indices, sentinel_indices, -1)
        if self.n_problems % self.batch_size != 0:
            mod_indices = np.append(mod_indices, -1)

        with self.s_indices.get_lock():
            self.s_indices[:] = mod_indices[:]
        with self.s_idx.get_lock():
            self.s_idx.value = -1

        del mod_indices, indices, sentinel_indices

    def current_batch_size(self, batch: int):
        if batch <= self.full_batches:
            return self.batch_size
        else:
            return self.remainder_batch_size

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

        if "test" in str(path) or "valid" in str(path):
            batch_size = len(problems)
        else:
            batch_size = args.batch_size

        indexer = mp.Value("i", 0)

        full_batches = len(problems) // batch_size
        if len(problems) % batch_size != 0:
            total_batches = full_batches + 1
        else:
            total_batches = full_batches
        # for sentinels to signal end of batch
        indices = mp.Array("i", len(problems) + total_batches)

        loader = ArrayLoader(
            problems,
            indices,
            indexer,
            batch_size=batch_size,
            full_batches=full_batches,
            seed=args.seed,
        )
        return loader, pset_dict

    def __len__(self):
        return self.n_problems
