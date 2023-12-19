#!/bin/bash
#SBATCH --job-name=6_2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=output.log
#SBATCH --partition=exercise   # partition (queue)
# Define the array sizes you want to test
sizes=(1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)

# Loop over array sizes and run the program
for size in "${sizes[@]}"; do
    echo "Running with array size: $size"
    ./bin/reduction $size
done
