from pathlib import Path
import json
import old_stp as old_stp
import torch as to
from enums import FourDir
import src.domains.stp as new_stp
from src.models import ConvNetSingle
import numpy as np
import random


def test_compare_with_old():

    old_specs_path = Path("problems/stp/5w/50000-original.txt")
    new_specs_path = Path("problems/stp/5w/50000-original.json")
    model_path = None
    # model_path = Path(
    #     "trained_models/Witness_4w4c-50000-original_Levin_1_1673639173_forward.pt"
    # )

    # load old problems
    old_problems = []
    with old_specs_path.open("r") as file:
        problems = file.readlines()
        for i in range(len(problems)):
            puzzle = old_stp.SlidingTilePuzzle(problems[i])
            old_problems.append(puzzle)

    new_problemset_dict = json.load(new_specs_path.open("r"))
    (
        new_problems,
        num_actions,
        in_channels,
        state_t_width,
        double_backward,
    ) = new_stp.load_problemset(new_problemset_dict)

    assert len(new_problems) == len(old_problems)

    def state_equal(old, new):
        old_tiles = np.array(old._tiles)
        new_tiles = new[1].tiles.flatten()
        assert np.array_equal(old_tiles, new_tiles)
        return True

    def actions_equal(old, new):
        assert len(old_actions) == len(new_actions)
        for a in new_actions:
            assert a in old_actions
        for a in old_actions:
            assert a in new_actions
        return True

    # model = ConvNetSingle(25, 5, (2, 2), 32, 4)
    # model.load_state_dict(to.load(model_path))

    for old_prob, new_prob in zip(old_problems, new_problems):
        old_state = old_prob
        new_stp_domain = new_prob[1]
        new_state = new_stp_domain.reset()

        n = random.randint(2000, 5000)
        end = False
        for i in range(n):
            assert state_equal(old_state, (new_stp_domain, new_state))
            is_sol = old_state.is_solution()
            assert is_sol == new_stp_domain.is_goal(new_state)

            old_actions = old_state.successors()
            old_actions = [FourDir(a) for a in old_actions]
            new_actions = new_stp_domain.actions_unpruned(new_state)
            actions_equal(old_actions, new_actions)

            if is_sol:
                print(f"solution found for problem {new_prob[0]}")
                end = True
            if not old_actions:
                print(f"no actions problem {new_prob[0]}")
                end = True
            if end:
                print(f"checked {i} state/action pairs")
                break

            action = random.choice(old_actions)
            old_state.apply_action(action)
            new_state = new_stp_domain.result(new_state, action)


if __name__ == "__main__":
    test_compare_with_old()
