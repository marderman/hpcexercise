#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH -p exercise
#SBATCH -w ceg-brook01
#SBATCH -t 0-0:10                   # timelimit: (D-HH:MM)
#SBATCH --job-name=test_run         # job name, "OMP_run"
#SBATCH -o ex2_out.txt

blk_num_step=1000
thread_step=50
meas_repetitions=10

# echo "Fix blocks, variate threads"
echo "# Blocks, # Threads, Sync. time, Async. time"
for (( blks = 1; blks < 16384; blks += $blk_num_step )) ; do
	for (( thrs = 1; thrs <= 1024 ; thrs += $thread_step )) ; do
		for (( i = 0; i < 10 ; i++ )) ; do
			# echo "$i: Blks: $blks, thrs: $thrs"
			bin/nullKernelAsync $blks $thrs
		done
	done
done

# echo
# echo "Fix threads, variate blocks"
# for (( thrs = 1; thrs <= 1024 ; thrs += $thread_step )) ; do
# 	for (( blks = 1; blks < 16384; blks += $blk_num_step )) ; do
# 		for (( i = 0; i < 10 ; i++ )) ; do
# 			# echo "$i: Blks: $blks, thrs: $thrs"
# 			bin/nullKernelAsync $blks $thrs
# 		done
# 	done
# done
