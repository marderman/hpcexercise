#!/bin/bash

#SBATCH --job-name=test_run         # job name, "OMP_run"
#SBATCH --partition=excercise_hpc       # partition (queue)
#SBATCH -t 0-0:10                   # timelimit: (D-HH:MM)
#SBATCH  --nodes=1                  # number of nodes
#SBATCH --ntasks-per-node=8         # number of cores
#SBATCH --output=slurm.out          # file to collect standard output
#SBATCH --error=slurm.err           # file to collect standard errors

module load mpi/open-mpi-4.0.5
mpirun -np 3 hostname