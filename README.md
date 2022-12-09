# TODO

- Multiprocess same-gpu doesn't work, throws c10 throws an error during all_gather of the
  batch_results. Not that important since cpu seems to be much faster anyways.
- Gradients probably shouldn't always be scaled by world_size, since not every process contributes
  to the overall gradient (only the ones that solve a problem do).
- Look into OMP_NUM_TREADS tuning when on ComputeCanada.
- Fix relative/absolute imports, partircularly environment in slidting_tile_puzzle.py is senseitive
  to torchrun vs without

# Notes
- Order of problems seen is only deterministic for a given world_size.
- I compute the loss over a whole batch and do an update (repeated grad_steps times),
  in contrast to the original implementation doing an update for each trajectory in the bacth.
