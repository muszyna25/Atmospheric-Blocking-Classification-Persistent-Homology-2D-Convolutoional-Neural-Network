#!/bin/bash 

#-SBATCH -q debug
#SBATCH --qos premium
#SBATCH -N 1
#SBATCH -t 6:00
#SBATCH -J testing
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH --array=7-7

set -u 

module load tensorflow/intel-1.12.0-py36
pwd; date

echo $SLURM_ARRAY_TASK_ID

DATA_PATH="/global/cscratch1/sd/muszyng/ethz_data/testing_file/"
echo $DATA_PATH

FILE_IDX=$(( $SLURM_ARRAY_TASK_ID / 2 ))
echo $FILE_IDX
FILE_PART=$(( $SLURM_ARRAY_TASK_ID % 2 ))
echo $FILE_PART
#LINE=$(head -n $((${FILE_IDX})) | tail -n 1 "file_list.txt")
LINE=$(head -n 1 "file_list.txt")
echo $LINE
INPUT_FILE=${LINE}
N_SUB_IMGS=0

#srun -n 64 python task.py ${INPUT_FILE} ${FILE_PART}
srun -n 2 -l python task.py ${INPUT_FILE} ${FILE_PART} ${N_SUB_IMGS}

date

