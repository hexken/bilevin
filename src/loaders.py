import numpy as np

from search.utils import Problem


class ProblemLoader:
    def __init__(
        self,
        local_problems: list[list[Problem]],
        all_ids: list[str],
        shuffle: bool = True,
        rng=None,
        seed: int = 1,
        manual_advance: bool = False,
    ):
        if not rng:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

        self.shuffle = shuffle
        self.problems = local_problems
        self.all_ids = all_ids  # ids of problems accross all ranks
        self.n_stages = len(self.problems)
        self.manual_advance = manual_advance
        self.stage = -1

    def __iter__(self):
        self.goto_next_stage = True
        self.stage = -1
        return self

    def _advance_stage(self) -> bool:
        """Returns True if there are no more stages"""
        self.stage += 1
        if self.stage >= self.n_stages:
            return True
        probs = self.problems[self.stage]
        self.stage_problems = np.empty(len(probs), dtype=object)
        self.stage_problems[:] = probs
        if self.shuffle:
            self._indices = self.rng.permutation(len(self.stage_problems))
        else:
            self._indices = np.arange(len(self.stage_problems))
        self.stage_all_ids = self.all_ids[self.stage]
        self._idx = 0
        return False

    def __next__(self):
        if self.goto_next_stage:
            self.goto_next_stage = False
            done = self._advance_stage()
            if done:
                raise StopIteration
        problem = self.stage_problems[self._indices[self._idx]]
        self._idx += 1
        if self._idx == len(self.stage_problems):
            if not self.manual_advance:
                self.goto_next_stage = True
            else:
                if self.shuffle:
                    self._indices = self.rng.permutation(len(self.stage_problems))
                    self._idx = 0
        return problem
