#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

for model in ${masked_lm_models[@]}; do
    experiment_id="stereoset_m-${model}_c-${model_to_model_name_or_path[${model}]}"
    if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
        echo ${experiment_id}
        sbatch \
            --gres=gpu:1 \
            -J ${experiment_id} \
            -o ${persistent_dir}/logs/%x.%j.out \
            -e ${persistent_dir}/logs/%x.%j.err \
            python_job.sh experiments/stereoset.py \
                --model ${model} \
                --model_name_or_path ${model_to_model_name_or_path[${model}]}
    fi
done


for model in ${causal_lm_models[@]}; do
    experiment_id="stereoset_m-${model}_c-${model_to_model_name_or_path[${model}]}"
    if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
        echo ${experiment_id}
        sbatch \
            --gres=gpu:1 \
            -J ${experiment_id} \
            -o ${persistent_dir}/logs/%x.%j.out \
            -e ${persistent_dir}/logs/%x.%j.err \
            python_job.sh experiments/stereoset.py \
                --model ${model} \
                --model_name_or_path ${model_to_model_name_or_path[${model}]}
    fi
done
