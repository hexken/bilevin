# exp params
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

- try path "contractions"

# todo
- fix make_tables skipping when shouldnt
- fix plot wot owrk with keys and styles
- use total expanded for bidir loss?

# differences
- I don't eval duplicate states
- I mask out unavailable actions

# run differences
- ones with 4k initial loss are closer to levis, no masking
stp5-50000-train_PHS_1_36126099-1.out
stp5-50000-train_PHS_3_36126106-3.out

- ones with 1k initial loss are with masking
stp5-50000-train_PHS_1_36132336-1.out
stp5-50000-train_PHS_3_36132227-3.out

- masking lr 0.0005
stp5-50000-train_PHS_1_36179028-1.out
stp5-50000-train_PHS_3_36178869-3.out

-masking lr0.001
stp5-50000-train_PHS_1_36180020-1.out
stp5-50000-train_PHS_3_36179922-3.out



