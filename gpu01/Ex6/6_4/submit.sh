#!/usr/bin/env bash

#SBATCH --job-name=matMul
#SBATCH --partition=exercise   # partition (queue)
#SBATCH -t 0-01:00              # time limit: (D-HH:MM) 
#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH --error=slurm.err      # file to collect standard errors

module load devtoolset/10 cuda/11.6

make

# ./bin/reduction

for ((t = 2; t <= 1024; t*=2))
do
    for ((i = 128; i <= 50000000; i*=2))
    do
    ./bin/reduction -s $i -t $t
    done
done


# ./bin/reduction -s 16 -t 4