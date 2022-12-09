#!/bin/bash

salloc \
--nodes=1 \
--ntasks-per-node=48 \
--mem=187G \
--exclusive \
--account=def-lelis \
--time=00:30:00
