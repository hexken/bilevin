#!/bin/bash

    #mv -- "$file" "${file%%.old}"
for file in *; do
    mv -- "$file" "${file##cc_}"
done
