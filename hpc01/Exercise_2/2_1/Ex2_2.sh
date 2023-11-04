#!/bin/bash
ncores=$1
nruns=$2
cores_per_node=12
nodes=$(($ncores / ($cores_per_node+1)+1))
sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=mpi_job_hpc01
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --ntasks=${ncores}
#SBATCH --time=00:10:00
#SBATCH --nodes=${nodes}
#SBATCH --partition=exercise_hpc


module load devtoolset/10 mpi/open-mpi-4.1.6


for irun in "$(seq 1 ${nruns})"
do
    timestamp=\$(seq 1 ${nruns})
    srun  --distribution=block:block bin/ringCommunication | sort  >> "bench_${ncores}cores_run${irun}_${timestamp}.out"
done
EOF
