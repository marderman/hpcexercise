#!/bin/bash

max_cores=$1
runs=$2

for i in $(seq  2 2 $max_cores)
do
echo  $i
   sbatch ./Ex2_2cyclic.sh $i $runs
   sbatch ./Ex2_2block.sh $i $runs
done
