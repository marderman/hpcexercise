#!/usr/bin/env bash

#SBATCH --job-name=matMul
#SBATCH --partition=exercise   # partition (queue)
#SBATCH -t 0-01:00              # time limit: (D-HH:MM) 
#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH --error=slurm.err      # file to collect standard errors

module load devtoolset/10 cuda/11.4

make rebuild

#5.3.3
for ((size = 32; size <= 1024; size*=2))
do
    bin/matMul -s $size -t 32 --shared >> out_5_3_3.txt
done
    
# 5.3.2
# for ((threads = 2; threads <= 32; threads+=2))
#     do
#         bin/matMul -s 1024 -t $threads --shared >> out_5_3_2.txt
#     done