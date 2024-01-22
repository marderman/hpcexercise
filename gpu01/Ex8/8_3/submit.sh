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

echo "Vector Length;NumGangs;Grid Size;Total Alive Cells;Total Execution Time"


for ((j = 128; j <= 1024; j+=128))
do
    for ((k = 1024; k <= 10240; k+=1024))
    do
        for ((i = 128; i <= 32768; i*=2))
        do
            ./bin/gameoflife $i 1024 $k 
        done
    done
done


