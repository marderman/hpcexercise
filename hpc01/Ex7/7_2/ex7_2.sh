#!/bin/sh

#SBATCH --job-name=nBody
#SBATCH --partition=exercise_hpc   # partition (queue)
#SBATCH -t 0-01:00              # time limit: (D-HH:MM) 
#SBATCH --nodes=4              # number of nodes
#SBATCH --ntasks-per-node=12
#SBATCH --output=nBody.out     # file to collect standard output
#SBATCH --error=nBody.err      # file to collect standard errors

nodes=(1 2 3 4)
#nodes=(2)
# Define an array of tasks per node configurations (e.g., tasks_per_node=(4 8 16))
tasks_per_node=(2 4 8 12)
#tasks_per_node=( 2 )
rm data.txt

module load devtoolset/10 mpi/open-mpi-4.0.3
mpicc -o bin/nBody nBody.c -lm -o3

for node in "${nodes[@]}"; do
	echo "Node Count: ${node}" >> data.txt
        for tasks in "${tasks_per_node[@]}"; do
		echo "Tasks per Node ${tasks}, overall Tasks $((tasks*node))" >> data.txt
                for ((i = 128; i <= 32768; i*=2)); do
# echo "nBody $i" >> data.txt
                        srun --nodes=$node --ntasks $((tasks*node)) ./build/programm -b $i -n 100 >> data.txt
                done
        done
done
