from pathlib import Path
import json
import old_witness as old_wit
import src.domains.witness as new_wit
import src.domains as domains
import numpy as np
import random


def test_compare_with_old():

    old_specs_path = Path("problems/witness/original_50k_train.txt")
    new_specs_path = Path("problems/witness/4w4c/50000-original.json")

    old_problems = {}
    # load old problems
    with old_specs_path.open("r") as file:
        puzzle = file.readlines()
        j = 0
        i = 0
        while i < len(puzzle):
            k = i
            while k < len(puzzle) and puzzle[k] != "\n":
                k += 1
            s = old_wit.WitnessState()
            s.read_state_from_string(puzzle[i:k])
            old_problems["puzzle_" + str(j)] = s
            i = k + 1
            j += 1

    old_problems = list(old_problems.items())

    new_problemset_dict = json.load(new_specs_path.open("r"))
    (
        new_problems,
        num_actions,
        in_channels,
        state_t_width,
        double_backward,
    ) = new_wit.load_problemset(new_problemset_dict)

    assert len(new_problems) == len(old_problems)

    for i in range(len(new_problems)):
        old_prob = old_problems[i]
        new_prob = new_problems[i]

        assert new_prob[0] == int(old_prob[0].split("_")[1])

        old_state = old_prob[1]

        new_wit_domain = new_prob[1]
        new_state = new_wit_domain.reset()

        def state_equals(old, new):
            assert np.array_equal(old._dots, new.grid)
            assert np.array_equal(old._v_seg, new.v_segs)
            assert np.array_equal(old._h_seg, new.h_segs)
            return True

        n = random.randint(2000, 5000)
        for i in range(n):
            assert state_equals(old_state, new_state)
            assert old_state.has_tip_reached_goal() == new_wit_domain.is_head_at_goal(
                new_state
            )
            assert old_state.is_solution() == new_wit_domain.is_goal(new_state)

            old_actions = old_state.successors()
            new_actions = new_wit_domain.actions_unpruned(new_state)
            assert len(old_actions) == len(new_actions)
            for a in new_actions:
                assert a in old_actions
            for a in old_actions:
                assert a in new_actions

            if old_actions:
                action = random.choice(old_actions)
                old_state.apply_action(action)
                new_state = new_wit_domain.result(new_state, action)
            else:
                print("checked")

if __name__ == "__main__":
    test_compare_with_old()
