#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

# Actual time:   ["bert-base-uncased"]="00:20:00" ["albert-base-v2"]="00:20:00" ["roberta-base"]="00:20:00" ["gpt2"]="00:20:00"
declare -A time=(["bert-base-uncased"]="01:00:00" ["albert-base-v2"]="01:00:00" ["roberta-base"]="01:00:00" ["gpt2"]="01:00:00")

experiments="political"

for model in ${models[@]}; do
    for bias_type in ${bias_types[@]}; do
        for pre_assignment_type in ${pre_assignment_types[@]}; do
            experiment_id="projection_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}_s-0_p-${pre_assignment_type}"
            if [ ! -f "${persistent_dir}/results/projection_matrix/${experiment_id}.pt" ]; then
                echo ${experiment_id}
                sbatch \
                    --gres=gpu:1 \
                    --time ${time[${model_to_model_name_or_path[${model}]}]} \
                    -J ${experiment_id} \
                    -o ${persistent_dir}/logs/%x.%j.out \
                    -e ${persistent_dir}/logs/%x.%j.err \
                    python_job.sh experiments/inlp_projection_matrix.py \
                        --model ${model} \
                        --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                        --bias_type ${bias_type} \
                        --n_classifiers ${model_to_n_classifiers[${model}]} \
                        --seed 0 \
                        --pre_assignment_path "/bask/projects/j/jlxi8926-auto-sum/shun/code/unsup_svd/data/projection_matrix/${experiments}/${model}/${bias_type}_${pre_assignment_type}.npz"\
                        --pre_assignment_type ${pre_assignment_type}
            fi
        done
    done
done
