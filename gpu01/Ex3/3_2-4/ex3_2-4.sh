#!/usr/bin/env bash

#SBATCH --job-name=memCpyLeandro
#SBATCH --partition=exercise   # partition (queue)
#SBATCH -t 0-01:00              # time limit: (D-HH:MM) 
#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH --error=slurm.err      # file to collect standard errors

if [ -f "output_3_2_1.txt" ]; then
    rm "output_3_2_1.txt"
fi

if [ -f "output_3_2_2.txt" ]; then
    rm "output_3_2_2.txt"
fi

if [ -f "output_3_3.txt" ]; then
    rm "output_3_3.txt"
fi

if [ -f "output_3_4.txt" ]; then
    rm "output_3_4.txt"
fi



#3_2
for (( i=32; i<=1024; i*=2 ))
do 
    for (( c=1024; c<=1073741824; c*=1024 ))
    do 
        bin/memCpy --global-coalesced --t $i --g 1 --s $c >> output_3_2_1.txt
    done
done

for (( i=1; i<=32; i++ ))
do 
    for (( c=1024; c<=1073741824; c*=1024 ))
    do 
        bin/memCpy --global-coalesced --t 1024 --g $i --s $c >> output_3_2_2.txt
    done
done

#3_3
for i in {1..100}; do
    bin/memCpy --global-stride -t 1024 -g 32 --stride $i >> output_3_3.txt
done

#3_4
for i in {1..100}; do
    bin/memCpy --global-offset -t 1024 -g 32 --offset $i --i 100000 >> output_3_4.txt
done
