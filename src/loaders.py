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
from math import ceil

import numpy as np

from domains.domain import Problem


class ProblemsBatchLoader:
    def __init__(
        self,
        problems: list[Problem],
        all_ids: list[str],
        batch_size: int,
        world_size: int = 1,
        epochs: int = 1,
        shuffle: bool = True,
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
        self.problems[:] = problems
        self.all_ids = all_ids  # ids of problems accross all ranks
        self.batch_size = batch_size
        self._len = len(problems)
        self._num_problems_served = 0

        self.world_size = world_size
        self.world_num_batches = ceil(len(all_ids) / (batch_size * world_size))
        self.batches_served = 0

    def __len__(self):
        return self._len

    def __iter__(self):
        self.batches_served = 0
        if self.shuffle:
            self._indices = self.rng.permutation(self._len)
        else:
            self._indices = np.arange(self._len)

        self._num_problems_served = 0

        return self

    def __next__(self):
        if self._num_problems_served >= self._len:
            if self.batches_served < self.world_num_batches:
                self.batches_served += 1
                return []
            raise StopIteration
        # elif self._num_problems_served + self.batch_size > self._len:
        #     next_indices = self._indices[self._num_problems_served :]
        next_indices = self._indices[
            self._num_problems_served : self._num_problems_served + self.batch_size
        ]
        self._num_problems_served += len(next_indices)
        self.batches_served += 1
        return self.problems[next_indices]

    def __getitem__(self, idx):
        return self.problems[idx]


class CurriculumLoader:
    def __init__(
        self,
        bootstrap_problems: list[Problem],
        all_bootstrap_ids: list[str],
        bootstrap_epochs: int,
        curriculum: list[int],
        problems_per_difficulty: int,
        curriculum_problems: list[Problem],
        all_curriculum_ids: list[str],
        curriculum_epochs: int,
        permutation_problems: list[Problem],
        all_permutation_ids: list[str],
        permutation_epochs: int,
        batch_size: int,
        world_size: int,
        include_prev_difficulty: bool,
        seed: int = 1,
        shuffle: bool = True,
    ):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.all_bootstrap_ids = all_bootstrap_ids

        self.bootstrap_problems = bootstrap_problems
        self.bootstrap_epochs = bootstrap_epochs

        self.curriculum = curriculum
        self.num_curriculum_stages = len(curriculum)
        self.curriculum_problems = curriculum_problems
        self.all_curriculum_ids = all_curriculum_ids
        self.problems_per_difficulty = problems_per_difficulty
        self.curriculum_epochs = curriculum_epochs

        self.permutation_problems = permutation_problems
        self.all_permutation_ids = all_permutation_ids
        self.permutation_epochs = permutation_epochs
        self.world_ppd = problems_per_difficulty * world_size
        self.world_size = world_size
        self.include_prev_difficulty = include_prev_difficulty

    def __iter__(self):
        self.next_stage = "bootstrap"
        return self

    def __next__(self):
        if self.next_stage == "bootstrap":
            self.next_stage = "curriculum"
            self.curriculum_stage = -1
            self.stage = "bootstrap"
            self.problems = copy(self.bootstrap_problems)
            self.ids = copy(self.all_bootstrap_ids)
            self.loader = ProblemsBatchLoader(
                self.problems,
                self.ids,
                self.batch_size,
                self.world_size,
                self.bootstrap_epochs,
                False,  # don't shuffle the bootstrap loader on first epoch
                self.rng,
            )
        elif "curriculum" in self.next_stage:
            self.curriculum_stage += 1
            if self.curriculum_stage == self.num_curriculum_stages - 1:
                self.next_stage = "permutation"
            self.stage = f"curriculum_{self.curriculum_stage}"
            new_problems = self.curriculum_problems[
                (self.curriculum_stage * self.problems_per_difficulty) : (
                    self.curriculum_stage + 1
                )
                * self.problems_per_difficulty
            ]
            new_ids = self.all_curriculum_ids[
                (self.curriculum_stage * self.world_ppd) : (self.curriculum_stage + 1)
                * self.world_ppd
            ]
            if self.include_prev_difficulty:
                self.problems.extend(new_problems)
                self.ids.extend(new_ids)
            else:
                self.problems = new_problems
                self.ids = new_ids

            self.loader = ProblemsBatchLoader(
                self.problems,
                self.ids,
                self.batch_size,
                self.world_size,
                self.curriculum_epochs,
                self.shuffle,
                self.rng,
            )
        elif self.next_stage == "permutation":
            self.next_stage = "end"
            self.stage = "permutation"
            if self.include_prev_difficulty:
                self.problems.extend(self.permutation_problems)
                self.ids.extend(self.all_permutation_ids)
            else:
                self.problems = self.permutation_problems
                self.ids = self.all_permutation_ids
            self.loader = ProblemsBatchLoader(
                self.problems,
                self.ids,
                self.batch_size,
                self.world_size,
                self.permutation_epochs,
                self.shuffle,
                self.rng,
            )
        elif self.next_stage == "end":
            raise StopIteration

        return self.loader
