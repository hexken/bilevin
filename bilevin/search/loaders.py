from copy import deepcopy

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
        problems: list[list[Problem]],
        shared_inices,
        shared_indexer,
        batch_size=32,
        seed: int = 1,
    ):
        self.problems = problems[0]  # todo since no curriculum
        self.shared_indices = shared_inices
        self.shared_indexer = shared_indexer
        self.batch_size = batch_size

        self.rng = np.random.default_rng(seed)
        self.loaded_state = False
        self.n_problems = len(self.problems)

    def reset_indexer(self, shuffle: bool = False):
        if shuffle:
            new_indices = self.rng.permutation(self.n_problems)
        else:
            new_indices = np.arange(self.n_problems)

        with self.shared_indices.get_lock():
            self.shared_indices[:] = new_indices[:]
        with self.shared_indexer.get_lock():
            self.shared_indexer.value = 0

    def load_state(self, state: dict):
        with self.shared_indices.get_lock():
            self.shared_indices[:] = state["indices"][:]
        with self.shared_indexer.get_lock():
            self.shared_indexer.value = state["indexer"]
        self.batch_size = state["batch_size"]
        self.rng = state["rng"]

    def state_dict(self) -> dict:
        return {
            "indices": self.shared_indices[:],
            "indexer": self.shared_indexer.value,
            "batch_size": self.batch_size,
            "rng": self.rng,
        }

    def advance_batch(self):
        with self.shared_indexer.get_lock():
            idx = self.shared_indexer.value
            problem = self.problems[self.shared_indices[idx]]
            self.shared_indexer.value += 1
            return problem

    def get_problem(self):
        with self.shared_indexer.get_lock():
            idx = self.shared_indexer.value
            if idx == self.n_problems or idx % self.batch_size == 0:
                return None
            else:
                problem = self.problems[self.shared_indices[idx]]
                self.shared_indexer.value += 1
                return problem

    def __len__(self):
        return self.n_problems


class ProblemLoader:
    def __init__(
        self,
        world_num_problems: int,
        local_problems: list[list[Problem]],
        shuffle: bool = True,
        seed: int = 1,
    ):
        self.rng = np.random.default_rng(seed)
        self.shuffle = shuffle
        self.problems = local_problems
        self.n_stages = len(self.problems)
        self.stage = 0
        self.world_num_problems = world_num_problems
        self.loaded_state = False
        self.num_stages = len(self.problems)

    def __len__(self):
        return self.world_num_problems

    def __iter__(self):
        if not self.loaded_state:
            self.stage_complete = True
            self.repeat_stage = False
        return self

    def get_state(self):
        state = {
            "indices": self._indices,
            "_idx": self._idx,
            "stage": self.stage,
            "rng": self.rng,
            "stage_complete": self.stage_complete,
            "repeat_stage": self.repeat_stage,
        }
        return state

    def load_state(self, state):
        self._indices = state["indices"]
        self._idx = state["_idx"]
        self.stage = state["stage"]
        self.rng = state["rng"]
        self.stage_complete = state["stage_complete"]
        self.repeat_stage = state["repeat_stage"]
        self.loaded_state = True

        self.stage_problems = self.problems[self.stage - 1]

    def _advance_stage(self) -> bool:
        """Returns True if there are no more stages"""
        self.stage += 1
        if self.stage > self.n_stages:
            return True
        self.stage_problems = self.problems[self.stage - 1]
        if self.shuffle:
            self._indices = self.rng.permutation(len(self.stage_problems))
        else:
            self._indices = np.arange(len(self.stage_problems))
        self._idx = 0
        return False

    def __next__(self):
        if self.repeat_stage:
            self.stage_complete = False
            if self.shuffle:
                self._indices = self.rng.permutation(len(self.stage_problems))
                self._idx = 0
        elif self.stage_complete:
            self.stage_complete = False
            done = self._advance_stage()
            if done:
                raise StopIteration
        problem = self.stage_problems[self._indices[self._idx]]
        self._idx += 1
        if self._idx == len(self.stage_problems):
            if self.shuffle:
                self._indices = self.rng.permutation(len(self.stage_problems))
                self._idx = 0
        return deepcopy(problem)
