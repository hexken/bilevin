#!/bin/bash

    #mv -- "$file" "${file%%.old}"
for file in tri6_*; do
    cp -- "$file" "../col/col6_${file##tri6_}"
done
