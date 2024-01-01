#!/usr/bin/env bash

#SBATCH --job-name=nbodyGPU
#SBATCH --partition=exercise   # partition (queue)
#SBATCH -t 0-01:00              # time limit: (D-HH:MM) 
#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH --error=slurm.err      # file to collect standard errors

module load devtoolset/10 cuda/11.4

make

for ((i = 1024; i <= 1024*1024; i*=2))
do
    ./bin/nbody -s $i
done

