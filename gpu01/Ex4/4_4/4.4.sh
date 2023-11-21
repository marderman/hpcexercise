#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -p exercise
#SBATCH -o out/runtime.csv

srun ./4-4