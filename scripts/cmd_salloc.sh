#!/bin/bash

salloc \
--nodes=1 \
--ntasks-per-node=40 \
--mem=186G \
--exclusive \
--account=def-lelis \
--time=00:30:00
