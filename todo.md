- double check why LTS sucks on stp3
- fix witness envs
- regen all problemsets, retest
- retest checkpointing/test
- should we prioritize backward model when costs are equal? as we are now?
- Maybe we should actually use expanding a common state as the stopping condition, and report the
  plp of this expanded state, instead of using a common generates state. This is more in-line with
  the analysis, though maybe we can do analysis using the generated also?
- should we compute probability of the start/goal nodes? instead of considering them 1?
- expand might be causing issues
- ensure h is set!

- stp4c
- astar - lr 0.001 w2.5/3
- biastar - they all suck, choose one with good loss?
- bilevin  - lr 0.001
- levin - lr 0.001

col4
- astar - lr 0.001 w1/w1.5
- biastar - lr 0.001 w1, lr 0.1
- bilevin- lr 0.001
- levin - lr 0.001

tri4
- astar - lr 0.001 w2.5
- biastar - lr 0.001 w2.5, or l3 0.01 all w
- bilevin - lr 0.001
- levin - lr 0.001

