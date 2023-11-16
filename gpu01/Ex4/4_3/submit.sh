#!/usr/bin/env bash

#SBATCH --job-name=shConfl
#SBATCH --partition=exercise   # partition (queue)
#SBATCH -t 0-03:00             # time limit: (D-HH:MM)
#SBATCH --gres=gpu:1
#SBATCH --out=slurm.out
#SBATCH --error=slurm.err

meas_repetitions=10

echo "Block_dim,Stride,Clocks"
for (( thrs = 32; thrs <= 384; thrs *= 2 )) ; do
	for (( stride = 1; stride <= 32; stride += 1 )) ; do
		for (( i = 0; i < $meas_repetitions ; i++ )) ; do
			bin/memCpy --shared2register_conflict -g 99999999999 -t $thrs -stride $stride
		done
	done
done
