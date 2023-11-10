#!/bin/bash

#SBATCH --job-name=matrixMultiplySLeandro
#SBATCH --partition=exercise_hpc   # partition (queue)
#SBATCH -t 0-01:20              # time limit: (D-HH:MM) 
#SBATCH --nodes=1             # number of nodes
#SBATCH --ntasks-per-node=1   # number of cores
#SBATCH --output=slurm.out     # file to collect standard output
#SBATCH --error=slurm.err      # file to collect standard errors

srun ./bin/matrixMultiplyS