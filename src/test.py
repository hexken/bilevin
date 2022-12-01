import time


def test(initial_states, planner, model, time_limit_seconds):
    """ """
    solutions = {}

    for problem_name, initial_state in initial_states.items():
        initial_state.reset()

        solution_depth, expanded, generated, running_time = planner.search(
            initial_state, problem_name, -1, time.time(), time_limit_seconds, 0, model
        )

        solutions[problem_name] = (solution_depth, expanded, generated, running_time)

    for problem_name, data in solutions.items():
        print(
            "{:s}, {:d}, {:d}, {:d}, {:.2f}".format(
                problem_name, data[0], data[1], data[2], data[3]
            )
        )


def test_time_limit(
    initial_states, planner, model, time_limit_seconds, search_budget=-1
):
    """ """
    solutions = {}

    for problem_name, initial_state in initial_states.items():
        initial_state.reset()
        solutions[problem_name] = (-1, -1, -1, -1)

    start_time = time.time()

    while len(initial_states) > 0:

        for problem_name, initial_state in initial_states.items():
            solution_depth, expanded, generated, running_time = planner.search(
                initial_state,
                problem_name,
                search_budget,
                start_time,
                time_limit_seconds,
                model,
            )

            if solution_depth > 0:
                solutions[problem_name] = (
                    solution_depth,
                    expanded,
                    generated,
                    running_time,
                )
                del initial_states[problem_name]

        if (
            time.time() - start_time > time_limit_seconds
            or len(initial_states) == 0
            or search_budget >= 1000000
        ):
            for problem_name, data in solutions.items():
                print(
                    "{:s}, {:d}, {:d}, {:d}, {:.2f}".format(
                        problem_name, data[0], data[1], data[2], data[3]
                    )
                )
            return

        search_budget *= 2
