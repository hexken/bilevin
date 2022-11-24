#!/bin/bash

python src/main.py \
    --mode train \
    --algorithm Levin \
    --loss levin_loss \
    --model-path testpath/testmodel.pt \
    --domain SlidingTile \
    --problems-path problems/stp_test/3x3_1000/ \
    --initial-budget 7000 \
    --grad-steps 10 \
    --batch-size-expansions 32 \
    --batch-size-bootstrap 32\
