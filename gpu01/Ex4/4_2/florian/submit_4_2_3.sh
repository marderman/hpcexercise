#!/usr/bin/env bash

#SBATCH --job-name=SMem_4_2
#SBATCH --partition=exercise   # partition (queue)
#SBATCH -t 0-0:10              # time limit: (D-HH:MM) 
#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH --error=slurm.err      # file to collect standard errors

module load devtoolset/10 cuda/11.4


for ((blocks=1; blocks<=1024 ; blocks*=2))
do
    for ((i=32; i<=1024; i+=32))
    do
        srun bin/programm -s $((1024*48)) -g $blocks -t $i -i 10000 --shared2global >> centralized_out4_2_3.txt
    done
done

for ((blocks=1; blocks<=1024 ; blocks*=2))
do
    for ((i=32; i<=1024; i+=32))
    do
        srun bin/programm -s $((1024*48)) -g $blocks -t $i -i 10000 --global2shared >> centralized_out4_2_4.txt
    done
done