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
        self._len = len(self.all_ids)
        self.n_stages = len(self.problems)
        self.manual_advance = manual_advance

    def __iter__(self):
        self.stage = -1
        self._goto_next_stage = False
        self._advance_stage()
        if self.shuffle:
            self._indices = self.rng.permutation(len(self.stage_problems))
        else:
            self._indices = np.arange(len(self.stage_problems))
        return self

    def _advance_stage(self) -> bool:
        """Returns True if there are no more stages"""
        self.stage += 1
        if self.stage >= self.n_stages:
            return True
        probs = self.problems[self.stage]
        self.stage_problems = np.empty(len(probs), dtype=object)
        self.stage_problems[:] = probs
        self.stage_all_ids = self.all_ids[self.stage]
        self._idx = 0
        return False

    def __next__(self):
        if self._goto_next_stage:
            self.goto_next_stage = False
            done = self._advance_stage()
            if done:
                raise StopIteration
        if self._idx >= len(self.stage_problems):
            if not self.manual_advance:
                done = self._advance_stage()
                if done:
                    raise StopIteration
            else:
                if self.shuffle:
                    self._indices = self.rng.permutation(len(self.stage_problems))
        problem = self.stage_problems[self._indices[self._idx]]
        self._idx += 1
        return problem

    def stage_complete(self):
        return self._idx >= len(self.stage_problems)

    def manual_advance_stage(self):
        self._goto_next_stage = True
