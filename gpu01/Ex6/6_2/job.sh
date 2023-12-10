#!/bin/bash
#SBATCH --job-name=6_2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=output.log
#SBATCH --partition=exercise   # partition (queue)
# Define the array sizes you want to test
sizes=(128 256 512 1024 2048 4096)

# Loop over array sizes and run the program
for size in "${sizes[@]}"; do
    echo "Running with array size: $size"
    ./bin/reduction $size
done
