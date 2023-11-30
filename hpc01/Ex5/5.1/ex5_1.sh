#!/bin/bash

#SBATCH --job-name=heatMap
#SBATCH --partition=exercise_hpc   # partition (queue)
#SBATCH -t 0-01:00              # time limit: (D-HH:MM) 
#SBATCH --nodes=1             # number of nodes
#SBATCH --ntasks-per-node=12   # number of cores
#SBATCH --output=slurm.out     # file to collect standard output
#SBATCH --error=slurm.err      # file to collect standard errors

srun ./build/programm