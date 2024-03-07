#!/bin/bash

    #mv -- "$file" "${file%%.old}"
for f in *; do
    if [[ -d $f ]]; then
        cd $f
        for file in *; do
           if [[ $file == *.sh ]]; then
               cp -- "$file" "${file%.sh}_test.sh"
           fi
        done
        cd ..
    fi
done
               # echo "Renaming $file to ${file%.sh}_test.sh"
