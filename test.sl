#!/bin/bash
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --constraint=haswell
#SBATCH --time=1
#SBATCH --array=0-2

echo $SLURM_ARRAY_TASK_ID
