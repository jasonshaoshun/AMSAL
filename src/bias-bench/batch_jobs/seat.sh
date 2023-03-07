#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

for model in ${models[@]}; do
    experiment_id="seat_m-${model}_c-${model_to_model_name_or_path[${model}]}"
    if [ ! -f "${persistent_dir}/results/seat/${experiment_id}.json" ]; then
        echo ${experiment_id}
        sbatch \
            --gres=gpu:1 \
            -J ${experiment_id} \
            -o ${persistent_dir}/logs/%x.%j.out \
            -e ${persistent_dir}/logs/%x.%j.err \
            python_job.sh experiments/seat.py \
                --tests ${seat_tests} \
                --model ${model} \
                --model_name_or_path ${model_to_model_name_or_path[${model}]}
    fi
done
