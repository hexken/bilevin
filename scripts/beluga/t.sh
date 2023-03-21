#!/bin/bash

nlines=$(wc -l args.txt | cut -d" " -f1)
for i in $(seq 1 $nlines); do
    args=$(sed "${i}q;d" args.txt)
    echo $args
done

