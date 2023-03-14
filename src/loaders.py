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

from copy import copy

import numpy as np

from domains.domain import Problem


class ProblemsBatchLoader:
    def __init__(
        self,
        problems: list[Problem],
        all_ids: list[str],
        batch_size: int,
        epochs: int = 1,
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
        self.epochs = epochs
        self.problems = np.empty(len(problems), dtype=object)
        self.all_ids = all_ids
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
        # elif self._num_problems_served + self.batch_size > self._len:
        #     next_indices = self._indices[self._num_problems_served :]
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
        all_bootstrap_ids: list[str],
        bootstrap_dummy_last: bool,
        bootstrap_epochs: int,
        curriculum: list[int],
        problems_per_difficulty: int,
        curriculum_problems: list[Problem],
        all_curriculum_ids: list[str],
        curriculum_dummy_last: bool,
        curriculum_epochs: int,
        permutation_problems: list[Problem],
        all_permutation_ids: list[str],
        permutation_dummy_last: bool,
        permutation_epochs: int,
        batch_size: int,
        seed: int = 1,
        shuffle: bool = True,
    ):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.all_bootstrap_ids = all_bootstrap_ids

        self.bootstrap_problems = bootstrap_problems
        self.bootstrap_dummy_last = bootstrap_dummy_last
        self.bootstrap_epochs = bootstrap_epochs

        self.curriculum = curriculum
        self.curriculum_stages = len(curriculum)
        self.curriculum_problems = curriculum_problems
        self.all_curriculum_ids = all_curriculum_ids
        self.curriculum_dummy_last = curriculum_dummy_last
        self.problems_per_difficulty = problems_per_difficulty
        self.curriculum_epochs = curriculum_epochs

        self.permutation_problems = permutation_problems
        self.all_permutation_ids = all_permutation_ids
        self.permutation_dummy_last = permutation_dummy_last
        self.permutation_epochs = permutation_epochs

    def __iter__(self):
        self.stage = "bootstrap"
        self.stage_epoch = 0

        return self

    def __next__(self):
        self.stage_epoch += 1

        if self.stage == "bootstrap":
            if self.stage_epoch == 1:
                self.problems = copy(self.bootstrap_problems)
                self.ids = copy(self.all_bootstrap_ids)
                self.loader = ProblemsBatchLoader(
                    self.problems,
                    self.all_bootstrap_ids,
                    self.batch_size,
                    self.bootstrap_epochs,
                    False,  # don't shuffle the bootstrap loader on first epoch
                    self.bootstrap_dummy_last,
                    self.rng,
                )
            elif self.stage_epoch <= self.bootstrap_epochs:
                # do not update loader
                self.loader.shuffle = self.shuffle
            else:
                self.curriculum_difficulty = 0
                self.stage = "curriculum"
                self.stage_epoch = 1

        if "curriculum" in self.stage:

            if self.stage_epoch > self.curriculum_epochs:
                self.curriculum_difficulty += 1
                self.stage_epoch = 1
            if self.curriculum_difficulty >= self.curriculum_stages:
                self.stage = "permutation"
            elif self.stage_epoch == 1:
                self.stage = f"curriculum_{self.curriculum_difficulty}"
                self.problems.extend(
                    self.curriculum_problems[
                        (self.curriculum_difficulty * self.problems_per_difficulty) : (
                            self.curriculum_difficulty + 1
                        )
                        * self.problems_per_difficulty
                    ]
                )
                self.ids.extend(
                    self.all_curriculum_ids[
                        (self.curriculum_difficulty * self.problems_per_difficulty) : (
                            self.curriculum_difficulty + 1
                        )
                        * self.problems_per_difficulty
                    ]
                )
                self.loader = ProblemsBatchLoader(
                    self.problems,
                    self.ids,
                    self.batch_size,
                    self.curriculum_epochs,
                    self.shuffle,
                    self.curriculum_dummy_last,
                    self.rng,
                )

        if self.stage == "permutation":
            if self.stage_epoch == 1:
                self.problems.extend(self.permutation_problems)
                self.ids.extend(self.all_permutation_ids)
                self.loader = ProblemsBatchLoader(
                    self.problems,
                    self.all_permutation_ids,
                    self.batch_size,
                    self.permutation_epochs,
                    self.shuffle,
                    self.permutation_dummy_last,
                    self.rng,
                )
            elif self.stage_epoch > self.permutation_epochs:
                raise StopIteration

        return self.loader
