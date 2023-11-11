#!/usr/bin/env bash

#SBATCH --job-name=memCpyWalli
#SBATCH --partition=exercise   # partition (queue)
#SBATCH -t 0-6:00              # time limit: (D-HH:MM)
#SBATCH --gres=gpu:1
#SBATCH -w ceg-brook01
#SBATCH -o slurm.out
#SBATCH --error=slurm.err      # file to collect standard errors

linspace() {
    local start="$1"
    local stop="$2"
    local num_points="$3"

    if (( num_points <= 1 )); then
        echo "$start"
        return
    fi

    local step=$(bc -l <<< "($stop - $start) / ($num_points - 1)")
    for ((i = 0; i < num_points; i++)); do
	printf "%.0f\n" "$(bc -l <<< "$start + $i * $step")"
    done
}

logspace() {
    local start="$1"
    local stop="$2"
    local num_points="$3"
    local base="$4"  # Added argument for the base of the logarithm

    if (( num_points <= 1 )); then
        echo "$start"
        return
    fi

    local log_start=$(awk "BEGIN {print log($start) / log($base)}")
    local log_stop=$(awk "BEGIN {print log($stop) / log($base)}")
    local step=$(bc -l <<< "($log_stop - $log_start) / ($num_points - 1)")

    values=()
    for ((i = 0; i < num_points; i++)); do
        value=$(bc -l <<< "$base ^ ($log_start + $i * $step)" 2>/dev/null)
        values+=("$value")
    done

    # Sort the values
    sorted_values=($(printf "%s\n" "${values[@]}" | sort -n | uniq))

    # Print the sorted values
    for value in "${sorted_values[@]}"; do
        echo "$value"
    done
}

min_pkt_size=1024
max_pkt_size=$(( 1024**3 ))

min_tpb=32
max_tpb=1024

tpbs_num=5
pkt_sizes_num=3

echo "Host_memory type,Memory check status, Copy type, Size [B], # of Blocks, # of Threads per Block, Throughput [GBps]"

tpbs=$(linspace $min_tpb $max_tpb $tpbs_num)
pkt_sizes=$(logspace $min_pkt_size $max_pkt_size $pkt_sizes_num 2)

for size_val in ${pkt_sizes[@]} ; do
    for tpb_val in ${tpbs[@]} ; do
        # echo "Size: $size_val, TBP: $tpb_val"
        bin/memCpy -s $size_val -t $tpb_val -g 1 --global-coalesced
    done
done
