#!/bin/bash
#SBATCH -A acf-utk0011  #Write your project account associated to utia condo
#SBATCH -p short
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1  #--ntasks-per-node is used when we want to define the number of processes per node
#SBATCH --cpus-per-task=16
#SBATCH -o hello_world.o%j  
#SBATCH --qos=short
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

###########   Run your parallel executable with srun   ###############
srun ./hybrid #-n is the total number of processes for the job
