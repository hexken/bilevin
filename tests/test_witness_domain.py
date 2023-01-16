from pathlib import Path
import json
import old_witness as old_wit
import src.domains.witness as new_wit
import src.domains as domains


def test_compare_with_old():

    old_specs_path = Path("problems/witness/original_50k_train.txt")
    new_specs_path = Path("problems/witness/4w4w/50000-original.json")

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

    for i in range( len(new_problems)):
        old_prob = old_problems[i]
        new_prob = new_problems[i]

        assert new_prob[0] == old_prob[0] # same id

