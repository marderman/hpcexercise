#!/usr/bin/env bash

#SBATCH --job-name=nbodyGPU
#SBATCH --partition=exercise   # partition (queue)
#SBATCH -t 0-01:00              # time limit: (D-HH:MM) 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=12      # Number of tasks (tasks = processes in MPI/OpenMP)
#SBATCH -o slurm.out
#SBATCH --error=slurm.err      # file to collect standard errors

module load devtoolset/10 cuda/11.6 nvhpc/21.9

make
    
echo "Grid Size;Total Alive Cells;Total Execution Time"

for ((i = 128; i <= 8192; i*=2))
do
    ./bin/gameoflife $i
done
