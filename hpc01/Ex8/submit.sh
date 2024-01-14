#!/bin/sh

#SBATCH --job-name=trainCNN
#SBATCH --partition=exercise_hpc   # partition (queue)
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH -t 0-02:00             # time limit: (D-HH:MM)
#SBATCH --output=slurm.out     # file to collect standard output
#SBATCH --error=slurm.err      # file to collect standard errors

python cifr10_tutorial.py
