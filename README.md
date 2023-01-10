# TODO

- Update sokoban/sokoban problem generator to match new API
- Update stp/sokoban to use json
- Impl other bidir search algos?
- Try out torchscript

# Notes
- I compute the loss over a whole batch (by creating a MergedTrajectory) and do an update (repeated grad_steps times).
- No point in shuffling the MergedTrajectory since we treat it as a batch anyways.
- Training should be deterministic for a given world_size (with torch-deterministic set to true).
- Under this bootstrap training scheme (and same for the original implementation), the probability that the
  budget increases decreases as the number of problems increases, i.e. there is a trade-off between
  having to solve more problems and increasing the budget less frequently.
