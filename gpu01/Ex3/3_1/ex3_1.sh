#!/usr/bin/env bash

#SBATCH --job-name=copyMemLeandro
#SBATCH --partition=exercise   # partition (queue)
#SBATCH -t 0-0:10              # time limit: (D-HH:MM) 
#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH --error=slurm.err      # file to collect standard errors

bin/globalMemory