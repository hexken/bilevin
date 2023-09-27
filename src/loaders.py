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
from typing import Optional

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
        self.manual_advance = manual_advance
        self.problems = np.empty(len(local_problems), dtype=object)
        self.problems[:] = local_problems
        self.all_ids = all_ids  # ids of problems accross all ranks
        self._len = len(self.all_ids)
        self.n_stages = len(self.problems)

    def __len__(self):
        return self._len

    def __iter__(self):
        self.stage = -1
        self._advance_stage = False
        self._advance_stage(self.stage)
        return self

    def _advance_stage(self):
        self.stage_problems = self.problems[self.stage]
        self.stage_all_ids = self.all_ids[self.stage]
        if self.shuffle:
            self._indices = [self.rng.permutation(len(sp)) for sp in self.problems]
        else:
            self._indices = [np.arange(len(sp)) for sp in self.problems]

        self._served_this_stage = 0

    def __next__(self):
        if self._advance_stage:
            self._advance_stage = False
            self.stage += 1
            self._advance_stage(self.stage)
        if self._served_this_stage >= len(self.stage_problems):
            if not self.manual_advance:
                self._advance_stage = True
            problem = self.rng.choice(self.stage_problems)
        if self.stage >= self.n_stages:
            raise StopIteration
        return self.stage_problems[self._indices[self._served_this_stage]]

    def advance_stage(self):
        self._advance_stage = True
