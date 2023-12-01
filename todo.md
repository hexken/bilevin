- double check why LTS sucks on stp3
- impl a star losses
- fix witness envs
- regen all problemsets, retest
- retest checkpointing/test
- should we prioritize backward model when costs are equal? as we are now?
- Maybe we should actually use expanding a common state as the stopping condition, and report the
  plp of this expanded state, instead of using a common generates state. This is more in-line with
  the analysis, though maybe we can do analysis using the generated also?
- does Levi, in H-Levin, compute
- should we compute probability of the start/goal nodes? instead of considering them 1?
-double check LTS is using LC in PQ
