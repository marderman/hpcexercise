#!/usr/bin/env bash

#SBATCH --job-name=redVol
#SBATCH --partition=exercise   # partition (queue)
#SBATCH -t 0-01:00              # time limit: (D-HH:MM) 
#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH --error=slurm.err      # file to collect standard errors

sizes=(8 32 128 256 1024 2048 4096 8192)

echo "Array size,Block size,Parallel reduction BW,Sequential reduction BW"

for thrs in "${sizes[@]}" ; do
	for ((i = thrs; i <= 8192; i*=2)) ; do
		for ((j = 0; j < 10; j+=1)) ; do
			./bin/reduction -s $i -t $thrs
		done
	done
done
