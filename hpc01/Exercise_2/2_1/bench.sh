#!/bin/bash

max_cores=$1
runs=$2

for i in $(seq  2 2 $max_cores)
do
    sbatch ./Ex2_2.sh $i $runs
done