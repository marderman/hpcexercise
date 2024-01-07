#!/bin/bash
#SBATCH --job-name=67     # Job name
#SBATCH --output=cuda_job.out    # Output file
#SBATCH --error=cuda_job.err     # Error file
#SBATCH --partition=exercise          # Specify the GPU partition
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=1      # Number of tasks (tasks = processes in MPI/OpenMP)
#SBATCH --time=00:10:00          # Maximum runtime in HH:MM:SS format

# Load the CUDA module
module load cuda/11.4
module load devtoolset/10

# Compile the CUDA program
nvcc 7-4.cu -o 67

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    # Run the compiled executable
    ./67
else
    echo "Compilation failed. Please check your CUDA code."
fi