#!/bin/sh

#SBATCH --job-name=trainCNN
#SBATCH --partition=exercise   # partition (queue)
#SBATCH --ntasks-per-node=12
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -t 0-06:00             # time limit: (D-HH:MM)
#SBATCH --output=slurm.out     # file to collect standard output
#SBATCH --error=slurm.err      # file to collect standard errors

module load cuda/11.6
module load anaconda/3

echo "CPU training:"
python cifr10_tutorial.py
echo "GPU training:"
python cifr10_tutorial.py --cuda
