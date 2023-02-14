#!/bin/bash

touch args.txt

for dirname in ../../runs/wit_levin_runs/*; do
    tmp=${dirname#*avggrads_}
    expname=${tmp%_167*}
    modelpath=runs/$(basename $dirname)
    echo $modelpath $expname >> args.txt
done

