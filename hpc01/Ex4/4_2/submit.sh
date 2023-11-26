#!/bin/bash

#SBATCH --job-name=htRelaxW
#SBATCH --partition=exercise_hpc # partition (queue)
#SBATCH -t 0-0:20                # time limit: (D-HH:MM)
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=1      # number of cores
#SBATCH --output=slurm.out       # file to collect standard output
#SBATCH --error=slurm.err        # file to collect standard errors

# Sample array of integers
int_array=(128 512 1024 2048 4096)

# Get the length of the array
array_length=${#int_array[@]}

# Iterate through the array using indices
for ((i = 0; i < array_length; i++)); do
	element="${int_array[i]}"
	./bin/heatRelax $element 100
done
