#!/bin/bash 

#-SBATCH --qos premium
#SBATCH --qos debug
#SBATCH -N 1
#SBATCH -t 8:00
#SBATCH -J testing
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH --array=1-1

### Bash options ###
set -u

BATCH_START_TIME=$(date)
echo "[+] ------START TIME (ST): $BATCH_START_TIME------"

### Load python module ###
module load tensorflow/intel-1.12.0-py36
echo "[+] ------Python module loaded: "
python -V

### Workspace on CSCRATCH ###
WKDIR="/global/cscratch1/sd/muszyng/homol_out/$SLURM_ARRAY_TASK_ID"

mkdir -p ${WKDIR}

cp task.py file_slicing.py ${WKDIR}

cd ${WKDIR}

PWD=$(pwd)
echo "[+] ------Current working dir (CWD): $PWD------"

FILE_IDX=${SLURM_ARRAY_TASK_ID} 
echo "[+] ------FILE_ID: $FILE_IDX"

### Input file ###
INPUT_FILE="/global/cscratch1/sd/muszyng/ethz_data/ecmwf_download/batch_scripts/ERA_INTERIM_1980.nc"
echo "[+] ------INPUT_FILE: $INPUT_FILE"

### File slicing ###
#Change number of tasks to 2 when debugging/testing!!!!
N_PROC=32

python file_slicing.py ${INPUT_FILE} ${FILE_IDX} ${N_PROC} 

FILE_NAME=$(ls *.npy)

########## Parallel code ############

srun -n ${N_PROC} -l python task.py ${FILE_NAME}       

######### Sequential code ###########
BATCH_END_TIME=$(date)
echo "[+] ------END TIME (ET) $BATCH_END_TIME------"


