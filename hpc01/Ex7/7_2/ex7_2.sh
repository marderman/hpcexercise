#!/bin/sh

#SBATCH --job-name=nBody
#SBATCH --partition=exercise_hpc   # partition (queue)
#SBATCH -t 0-01:00             # time limit: (D-HH:MM)
#SBATCH --output=slurm.out     # file to collect standard output
#SBATCH --error=slurm.err      # file to collect standard errors

# 2 4 8 12 24 48 64
nodes=(1 1 1 1 2 4 8)
tasks_per_node=(2 4 8 12 12 12 8)

length=${#nodes[@]}

echo "# Processes,# Elements,Time [s]"

for (( j = 0; j < length; j++ )) ; do
        for (( i = 128; i <= 32768; i*=2 )) ; do
                srun --partition=exercise_hpc --nodes=${nodes[$j]} --ntasks-per-node=${tasks_per_node[$j]} ./bin/nBody -b $i -n 100
        done
done
