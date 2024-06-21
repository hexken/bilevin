#!/bin/bash

widths="3"
domains="cube"
agents="phs levin astar"

for domain in $domains; do
    for width in $widths; do
        for agent in $agents; do
            ./setdirs.sh lelis2/${domain}${width}/${agent} c
        done
    done
done


