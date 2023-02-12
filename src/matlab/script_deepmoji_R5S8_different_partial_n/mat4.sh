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
# twitter_short_sentiment

/home/y26/shared/matlab/R2022a/bin/matlab -nodisplay -r "clear;\
partial_supervision = true; partial_ratio = 0.3; run_kmeans = false; save_oracle = false;\
include_complex_Z = false; model_name = '0.8'; experiments_folder = 'deepmoji-different-partial-n'; experiments = 'ratio_on_race'; bias_name = 'race';\
relative_path_to_project = '../../'; dataset_path = fullfile('data/', experiments_folder, experiments, model_name, 'all.mat');\
save_partial_n = true; save_assignment_method = true; save_path = fullfile(relative_path_to_project, 'data/projection_matrix/', experiments_folder, experiments, model_name); \
iter = 50; epoch_num = 5; N_max = 30000;\
k=2; biases_num = 1; biases_dim = [0, 2]; complex_index = []; main; exit"
