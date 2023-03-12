import numpy as np

from domains.domain import Problem


class ProblemsBatchLoader:
    def __init__(
        self,
        problems: list[Problem],
        batch_size: int,
        shuffle: bool = True,
        dummy_last: bool = False,
        rng=None,
        seed: int = 1,
    ):
        if not rng:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

        self.shuffle = shuffle
        self.problems = np.empty(len(problems), dtype=object)
        self.problems[:] = problems
        self.batch_size = batch_size
        self._len = len(problems)
        self._num_problems_served = 0

        self.dummy_last = dummy_last
        self._dummy_served = False

    def __len__(self):
        return self._len

    def __iter__(self):
        self._dummy_served = False
        if self.shuffle:
            self._indices = self.rng.permutation(self._len)
        else:
            self._indices = np.arange(self._len)

        self._num_problems_served = 0

        return self

    def __next__(self):
        if self._num_problems_served >= self._len:
            if self.dummy_last and not self._dummy_served:
                self._dummy_served = True
                return []

            raise StopIteration
        next_indices = self._indices[
            self._num_problems_served : self._num_problems_served + self.batch_size
        ]
        self._num_problems_served += len(next_indices)
        return self.problems[next_indices]

    def __getitem__(self, idx):
        return self.problems[idx]


class CurriculumLoader:
    def __init__(
        self,
        bootstrap_problems: list[Problem],
        bootstrap_epochs: int,
        curriculum: list[int],
        problems_per_difficulty: int,
        curriculum_problems: list[Problem],
        curriculum_epochs: int,
        permutation_problems: list[Problem],
        permutation_epochs: int,
        batch_size: int,
        seed: int = 1,
        shuffle: bool = True,
        dummy_last: bool = False,
    ):
        self.shuffle = shuffle
        self.dummy_last = dummy_last
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.bootstrap_problems = bootstrap_problems
        self.bootstrap_epochs = bootstrap_epochs

        self.curriculum = curriculum
        self.curriculum_stages = len(curriculum)
        self.curriculum_problems = curriculum_problems
        self.problems_per_difficulty = problems_per_difficulty
        self.curriculum_epochs = curriculum_epochs

        self.permutation_problems = permutation_problems
        self.permutation_epochs = permutation_epochs

    def __iter__(self):
        self.stage = "bootstrap"
        self.stage_epoch = 1

        return self

    def __next__(self):
        if self.stage == "bootstrap":
            if self.stage_epoch == 1:
                self.loader = ProblemsBatchLoader(
                    self.bootstrap_problems,
                    self.batch_size,
                    False,  # don't shuffle the bootstrap loader on first epoch
                    self.dummy_last,
                    self.rng,
                )
                self.stage_epoch += 1
            elif self.stage_epoch <= self.bootstrap_epochs:
                # do not update loader
                self.loader.shuffle = self.shuffle
                self.stage_epoch += 1
            else:
                self.curriculum_difficuly = 0
                self.stage = f"curriculum_{self.curriculum_difficuly}"
                self.stage_epoch = 1

        if "curriculum" in self.stage:
            if self.curriculum_difficuly < self.curriculum_stages:
                if self.stage_epoch == 1:
                    problems = self.bootstrap_problems
                    problems.extend(
                        self.curriculum_problems[
                            : (self.curriculum_difficuly + 1)
                            * self.problems_per_difficulty
                        ]
                    )
                    self.loader = ProblemsBatchLoader(
                        problems,
                        self.batch_size,
                        self.shuffle,
                        self.dummy_last,
                        self.rng,
                    )
                    self.stage_epoch += 1
                elif self.stage_epoch <= self.curriculum_epochs:
                    self.stage_epoch += 1
                else:
                    self.curriculum_difficuly += 1
                    self.stage = f"curriculum_{self.curriculum_difficuly}"
                    self.stage_epoch = 1
            else:
                self.stage = "permutation"
                self.stage_epoch = 1

        if self.stage == "permutation":
            if self.stage_epoch == 1:
                problems = self.bootstrap_problems
                problems.extend(self.curriculum_problems)
                problems.extend(self.permutation_problems)
                self.loader = ProblemsBatchLoader(
                    problems,
                    self.batch_size,
                    self.shuffle,
                    self.dummy_last,
                    self.rng,
                )
                self.stage_epoch += 1
            elif self.stage_epoch <= self.permutation_epochs:
                self.stage_epoch += 1
            else:
                raise StopIteration

        return self.loader
