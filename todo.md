- should we prioritize backward model when costs are equal? as we are now?
- Maybe we should actually use expanding a common state as the stopping condition, and report the
  plp of this expanded state, instead of using a common generates state. This is more in-line with
  the analysis, though maybe we can do analysis using the generated also?
- regen pancake and cube3, test cc params

- stp4c
- astar - lr 0.001 w2.5/3
- biastar - they all suck, choose one with good loss?
- bilevin  - lr 0.001
- levin - lr 0.001

col4
- astar - lr 0.001 w1
- biastar - lr 0.001 w1
- bilevin- lr 0.001
- levin - lr 0.001

tri4
- astar - lr 0.001 w2.5
- biastar - lr 0.001 w2.5, or l3 0.01 all w
- bilevin - lr 0.001
- levin - lr 0.001

-pancake12
- astar w2.5
- biastar alt, w2.5. bfs sucks

- allow continuing for more epochs
- try separate feature net (again)
- try path "contractions"
- grad steps for each traj?


- verticla line when curr ends
- plot valid search/exp/len beside train

bug in process data...
- change exp names
- use args.json to group/sort
- save array job info or something in rundir, and slurm out name
