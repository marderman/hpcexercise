#!/usr/bin/env bash

#SBATCH --job-name=copyMemLeandro
#SBATCH --partition=exercise   # partition (queue)
#SBATCH -t 0-0:10              # time limit: (D-HH:MM) 
#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH --error=slurm.err      # file to collect standard errors

module load devtoolset/10 cuda/12.0

for j in {1..100}; do
    bin/memCpy --global-stride -t 1024 -g 32 --stride $j -i 10000
done
