#!/usr/bin/env bash

#SBATCH --job-name=matMul
#SBATCH --partition=exercise   # partition (queue)
#SBATCH -t 0-01:00              # time limit: (D-HH:MM) 
#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH --error=slurm.err      # file to collect standard errors

module load devtoolset/10 cuda/11.6

make

./bin/reduction #-s 16 -t 8


# for ((size = 32; size <= 1024; size*=2))
# do
#     srun ./bin/matMul -s $size -t 32 --shared >> out_5_3_3.txt
# done