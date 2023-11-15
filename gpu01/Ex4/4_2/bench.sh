#!/usr/bin/env bash

#SBATCH --job-name=sharedmemCpyMax
#SBATCH --partition=exercise   # partition (queue)
#SBATCH -t 0-01:00              # time limit: (D-HH:MM) 
#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH --error=slurm.err      # file to collect standard errors


module load devtoolset/10 cuda/11.4
rm centralized-*
for ((i=32; i<=1024; i+=32))
do
    for ((mem=8 ; mem<=48 ; mem+=8))
    do
        srun ./bin/memCpy -s $((1000*$mem)) -g 1 -t $i -i 100000 --global2shared >> centralized_out_g2s_4_2_1.txt
	    srun bin/memCpy -s $((1000*$mem)) -g 1 -t $i -i 100000 --shared2global >> centralized_out_s2g_4_2_1.txt
    done
done
