#!/bin/bash

#SBATCH --job-name=msBWWalli
#SBATCH --partition=exercise_hpc # partition (queue)
#SBATCH -t 0-0:20                # time limit: (D-HH:MM)
#SBATCH --nodes=2                # number of nodes
#SBATCH --ntasks-per-node=1      # number of cores
#SBATCH --output=slurm.out       # file to collect standard output
#SBATCH --error=slurm.err        # file to collect standard errors

# module load devtoolset/10 mpi/open-mpi-4.1.6
mpirun ./bin/measureBandwith
