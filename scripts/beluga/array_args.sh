#!/bin/bash

touch args.txt

for dirname in ../../runs/wit_valid/*; do
    tmp=${dirname#*Witness-}
    expname=${tmp%_167*}
    modelpath=runs/wit_valid/$(basename $dirname)

    if [[ "$modelpath" == *BiLevin* ]]; then
        agent="BiLevin"
    else	
        agent="Levin"
    fi
    echo $agent $modelpath $expname >> args.txt
done

