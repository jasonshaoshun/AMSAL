#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

experiments="political"

for model in ${sal_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        for pre_assignment_type in ${pre_assignment_types[@]}; do
            experiment_id="projection_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}_s-0"
            if [ ! -f "${persistent_dir}/results/projection_matrix/${experiment_id}.pt" ]; then
                echo ${experiment_id}
                sbatch \
                    --gres=gpu:1 \
                    -J ${experiment_id} \
                    -o ${persistent_dir}/logs/%x.%j.out \
                    -e ${persistent_dir}/logs/%x.%j.err \
                    python_job.sh experiments/sal_eigenvectors.py \
                        --model ${model} \
                        --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                        --bias_type ${bias_type} \
                        --persistent_dir ${persistent_dir} \
                        --pre_assignment_type ${pre_assignment_type} \
                        --pre_assignment_path "/bask/projects/j/jlxi8926-auto-sum/shun/code/unsup_svd/data/projection_matrix/${experiments}/${debiased_model_to_base_model[${model}]}/${bias_type}_${pre_assignment_type}.npz"
            fi
        done
    done
done