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


/home/y26/shared/matlab/R2022a/bin/matlab -nodisplay -r "clear; \
partial_supervision = true; model_name = 'Deepmoji'; experiments = 'concat_twitter_vulgar'; bias_name = 'gender'; \
relative_path_to_project = '../../'; dataset_path = fullfile('data/', experiments, model_name, strcat(model_name, '_', bias_name, '.mat')); \
iter = 50; epoch_num = 5; N_max = 30000; partial_n = 500; \
k=4; biases_num = 1; biases_dim = [0, 2]; include_complex_Z = false; complex_index = []; main_partialsup; exit"
