#!/usr/bin/env bash

#SBATCH --job-name=matMul
#SBATCH --partition=exercise   # partition (queue)
#SBATCH -t 0-01:00              # time limit: (D-HH:MM) 
#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH --error=slurm.err      # file to collect standard errors

module load devtoolset/10 cuda/11.4

make

bin/matMul

if [ -f "output_5_2_2.txt" ]; then
    rm "output_5_2_2.txt"
fi
  
for (( i=32; i<=8192; i*=2 ))
do     
echo "Matrix Size" $i >> output_5_2_2.txt
    for (( j=1; j<=32; j+=1 ))
    do 
        bin/matMul -s $i -t $j >> output_5_2_2.txt
    done
done