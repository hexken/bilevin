# TODO

- Reimpl sokoban/sokoban problem generator.
- Impl other bidir search algos?
- Make sure last batch data is logged (since we currently only log data from every 5th batch).
- Unify problems specs with a standard notation, for example JSON.
- fix problems parsed/loaded logging

# Notes
- I compute the loss over a whole batch (by creating a MergedTrajectory) and do an update (repeated grad_steps times).
- No point in shuffling the MergedTrajectory since we treat it as a batch anyways.
- Training should be deterministic for a given world_size (with torch-deterministic set to true).
- Under this bootstrap training scheme (and same for the original implementation), the probability that the
  budget increases decreases as the number of problems increases, i.e. there is a trade-off between
  having to solve more problems and increasing the budget less frequently. See run results.
- For the Witness, should the state repr differentiate between the colors, or only the relative
  differences? In the orignal impl it is only the differences that matter.
