#!/bin/bash
#SBATCH --job-name=6_2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=output.log
#SBATCH --partition=exercise   # partition (queue)
# Define the array sizes you want to test
sizes=(100000 500000 1000000 5000000 10000000 50000000 100000000 500000000 1000000000 5000000000 10000000000 50000000000)

# Loop over array sizes and run the program
for size in "${sizes[@]}"; do
    echo "Running with array size: $size"
    ./6_2 $size
done
