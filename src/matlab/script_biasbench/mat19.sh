#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=sdf-cs1
#SBATCH --partition=standard
#SBATCH --time=24:00:00

# echo "Running on host ${HOSTNAME}"
# echo "Using ${SLURM_NTASKS_PER_NODE} tasks per node"
# echo "Using ${SLURM_CPUS_PER_TASK} tasks per cpu"

experiments="benchbias"
model_name="RobertaModel"
bias_name="gender"


/home/y26/shared/matlab/R2022a/bin/matlab -nodisplay -r "clear; partial_supervision = false;\
model_name = '${model_name}'; experiments = '${experiments}'; bias_name = '${bias_name}';\
dataset_path = 'data/${experiments}/${model_name}/${bias_name}.mat'; \
iter = 50; epoch_num = 5; N_max = 10000; \
partial_n = 2000; k=2; main_unsup; exit"

