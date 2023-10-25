#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -p exercise
#SBATCH -o ex2_out.txt

bin/nullKernelAsync
