import numpy as np

from search.utils import Problem


class ProblemLoader:
    def __init__(
        self,
        world_num_problems: int,
        local_problems: list[list[Problem]],
        shuffle: bool = True,
        seed: int = 1,
        manual_advance: bool = False,
    ):
        self.rng = np.random.default_rng(seed)
        self.shuffle = shuffle
        self.problems = local_problems
        self.n_stages = len(self.problems)
        self.manual_advance = manual_advance
        self.stage = -1
        self.world_num_problems = world_num_problems
        self.state = None

    def __len__(self):
        return self.world_num_problems

    def __iter__(self):
        return self

    def __call__(self, rank, state=None):
        if state is not None:
            self.load_state(state, rank)
        else:
            self.goto_next_stage = True
            self.stage = -1
        return self

    def get_state(self):
        state = {
            "indices": self._indices,
            "_idx": self._idx,
            "stage": self.stage,
            "rng": self.rng,
            "goto_next_stage": self.goto_next_stage,
        }
        return state

    def load_state(self, state, rank):
        self._indices = state["indices"][rank]
        self._idx = state["_idx"]
        self.stage = state["stage"]
        self.rng = state["rng"]
        self.goto_next_stage = state["goto_next_stage"]

        probs: list[Problem] = self.problems[self.stage]
        self.stage_problems = np.empty(len(probs), dtype=object)
        self.stage_problems[:] = probs

    def _advance_stage(self) -> bool:
        """Returns True if there are no more stages"""
        self.stage += 1
        if self.stage >= self.n_stages:
            return True
        probs: list[Problem] = self.problems[self.stage]
        self.stage_problems = np.empty(len(probs), dtype=object)
        self.stage_problems[:] = probs
        if self.shuffle:
            self._indices = self.rng.permutation(len(self.stage_problems))
        else:
            self._indices = np.arange(len(self.stage_problems))
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
        assert isinstance(problem, Problem)
        return problem
