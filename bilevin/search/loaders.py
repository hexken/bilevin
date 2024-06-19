from copy import deepcopy

import numpy as np

from domains.domain import Domain

class Problem:
    def __init__(self, id: int, domain: Domain):
        self.id: int = id
        self.domain: Domain = domain

    def __hash__(self):
        return self.id.__hash__()

    def __eq__(self, other):
        return self.id == other.id

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
