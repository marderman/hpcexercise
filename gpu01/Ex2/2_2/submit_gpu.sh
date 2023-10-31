#!/bin/bash

#SBATCH --job-name=test_run         # job name, "OMP_run"
#SBATCH --partition=excercise       # partition (queue)
#SBATCH -t 0-0:10                   # timelimit: (D-HH:MM)
#SBATCH --output=slurm.out          # file to collect standard output
#SBATCH --error=slurm.err           # file to collect standard errors
#SBATCH --gres=gpu:1

module load devtoolset/9 cuda/12.0
./a.out