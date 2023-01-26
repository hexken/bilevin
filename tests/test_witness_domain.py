from pathlib import Path
import json
import old_witness as old_wit
import torch as to
import src.domains.witness as new_wit
from src.models import ConvNetSingle
import src.domains as domains
import numpy as np
import random


def test_compare_with_old():

    old_specs_path = Path("problems/witness/original_50k_train.txt")
    new_specs_path = Path("problems/witness/4w4c/50000-original.json")
    model_path = None
    model_path = Path(
        "trained_models/Witness-4w4c-50000-original_Levin-2000_1_1674083307_forward.pt/"
    )

    # load old problems
    old_problems = []
    with old_specs_path.open("r") as file:
        puzzle = file.readlines()
        i = 0
        while i < len(puzzle):
            k = i
            while k < len(puzzle) and puzzle[k] != "\n":
                k += 1
            s = old_wit.WitnessState()
            s.read_state_from_string(puzzle[i:k])
            old_problems.append(s)
            i = k + 1

    new_problemset_dict = json.load(new_specs_path.open("r"))
    (
        new_problems,
        num_actions,
        in_channels,
        state_t_width,
        double_backward,
    ) = new_wit.load_problemset(new_problemset_dict)

    assert len(new_problems) == len(old_problems)

    def state_equal(old, new):
        assert np.array_equal(old._dots, new[1].grid)
        assert np.array_equal(old._v_seg, new[1].v_segs)
        assert np.array_equal(old._h_seg, new[1].h_segs)
        assert np.array_equal(old._cells, new[0].cells)
        return True

    def actions_equal(old, new):
        assert len(old_actions) == len(new_actions)
        for a in new_actions:
            assert a in old_actions
        for a in old_actions:
            assert a in new_actions
        return True

    model = ConvNetSingle(9, 5, (2, 2), 32, 4)
    model.load_state_dict(to.load(model_path))

    correct_actions = [0, 2, 2, 2, 1, 2, 0, 0, 3, 3, 3, 3, 0, 0, 2, 2, 2, 2]
    for id, new_wit_domain in new_problems:

        old_state = old_problems[id]
        new_state = new_wit_domain.reset()

        n = random.randint(2000, 5000)
        end = False
        for i in range(n):
            if i == len(correct_actions) - 1:
                k=232
            assert state_equal(old_state, (new_wit_domain, new_state))
            assert old_state.has_tip_reached_goal() == new_wit_domain.is_head_at_goal(
                new_state
            )
            is_sol = old_state.is_solution()
            assert is_sol == new_wit_domain.is_goal(new_state)

            old_actions = old_state.successors()
            new_actions = new_wit_domain.actions_unpruned(new_state)
            actions_equal(old_actions, new_actions)

            if is_sol:
                print(f">>>solution found for problem {id}")
                end = True
            elif not old_actions:
                print(f"ran out of actions problem {id}")
                end = True
            if end:
                print(f"checked {i} state/action pairs")
                break

            # action = correct_actions[i]
            action = random.choice(old_actions)
            old_state.apply_action(action)
            new_state = new_wit_domain.result(new_state, action)


if __name__ == "__main__":
    test_compare_with_old()
