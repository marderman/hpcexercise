#!/usr/bin/env bash

#SBATCH --job-name=shConfl
#SBATCH --partition=exercise   # partition (queue)
#SBATCH -t 0-01:00             # time limit: (D-HH:MM)
#SBATCH --gres=gpu:1
#SBATCH --out=slurm.out
#SBATCH --error=slurm.err

blk_num_step=1000
thread_step=50
meas_repetitions=5

# echo "Fix blocks, variate threads"
echo "# Blocks, # Threads, Sync. time, Async. time"
for (( blks = 1; blks <= 16384; blks *= 2 )) ; do
	for (( thrs = 1; thrs <= 1024; thrs *= 2 )) ; do
		for (( i = 0; i < $meas_repetitions ; i++ )) ; do
			bin/memCpy --shared2register_conflict -g $blks -t $thrs
		done
	done
done
