#!/usr/bin/env bash

#SBATCH --job-name=memCpyLeandro
#SBATCH --partition=exercise   # partition (queue)
#SBATCH -t 0-01:00              # time limit: (D-HH:MM) 
#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH --error=slurm.err      # file to collect standard errors

module load cuda/11.4 devtoolset/10

for i in {1..48}; do
  echo "First Loop: $((i*1000)) KB"



done
