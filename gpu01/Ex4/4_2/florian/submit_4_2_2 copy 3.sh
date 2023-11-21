#!/usr/bin/env bash

#SBATCH --job-name=SMem_4_2
#SBATCH --partition=exercise   # partition (queue)
#SBATCH -t 0-0:10              # time limit: (D-HH:MM) 
#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH --error=slurm.err      # file to collect standard errors

module load devtoolset/10 cuda/11.4


for ((i=32; i<=1024; i+=32))
do
    for ((mem=8; mem<=48 ; mem+=8))
    do
        srun bin/programm -s $((1024*$mem)) -g 1 -t $i -i 10000 --shared2global >> centralized_out4_2_2.txt
    done
done