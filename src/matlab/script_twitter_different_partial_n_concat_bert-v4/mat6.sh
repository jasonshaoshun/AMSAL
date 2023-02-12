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


/home/y26/shared/matlab/R2022a/bin/matlab -nodisplay -r "clear;\
partial_supervision = true; partial_ratio = 0.5; run_kmeans = false; save_oracle = false;\
include_complex_Z = false; model_name = 'BertModel'; experiments = 'twitter-different-partial-n-concat-v4'; bias_name = 'age-gender';\
relative_path_to_project = '../../'; dataset_path = fullfile('data/', experiments, model_name, strcat(bias_name, '.mat'));\
save_partial_n = true; save_assignment_method = true; save_path = fullfile(relative_path_to_project, 'data/projection_matrix/', experiments, model_name); \
iter = 50; epoch_num = 5; N_max = 30000;\
k=4; biases_num = 2; biases_dim = [0, 3, 5]; complex_index = []; main; exit"
