# TODO

- use JSON to specify problems. Refactor problems/domains/states structures. A Problem should
  have attributes name and data. A domain has methods is_goal, actions, result, etc. A state should
  maintain a tensor representation and potentially some auxiliary data. See branch domain for a
  beginning implementation.

# Notes
- I compute the loss over a whole batch (by creating a MergedTrajectory) and do an update (repeated grad_steps times).
- No point in shuffling the MergedTrajectory since we treat it as a batch anyways.
- Training should be deterministic for a given world_size (with torch-deterministic set to true).
  in contrast to the original implementation doing an update for each trajectory in the bacth.
- Under this training scheme (and same for the original implementation), the probability that the
  budget increases decreases as the number of problems increases, i.e. there is a trade-off between
  having to solve more problems and increasing the budget less frequently. See run results.
