# TODO

- Multiprocess same-gpu doesn't work, throws c10 throws an error during all_gather of the
  batch_results. Not that important since we don't plan to use this parallelization scheme.
- use JSON to write problems to files. Refactor problems/domains/env structures.

# Notes
- Order of problems seen is only deterministic for a given world_size.
- I compute the loss over a whole batch and do an update (repeated grad_steps times),
  in contrast to the original implementation doing an update for each trajectory in the bacth.
- Under this training scheme (and same for the original implementation), the probability that the
  budget increases decreases as the number of problems increases, i.e. there is a trade-off between
  having to solve more problems and increasing the budget less frequently. See run results.
