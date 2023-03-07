#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

# Actual time:   ["bert-base-uncased"]="00:15:00" ["albert-base-v2"]="00:15:00" ["roberta-base"]="00:10:00" ["gpt2"]="00:15:00"
declare -A time=(["bert-base-uncased"]="01:00:00" ["albert-base-v2"]="01:00:00" ["roberta-base"]="01:00:00" ["gpt2"]="01:00:00")



for model in ${inlp_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        for pre_assignment_type in ${pre_assignment_types[@]}; do
            experiment_id="seat_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}_p-${pre_assignment_type}"
            if [ ! -f "${persistent_dir}/results/seat/${experiment_id}.json" ]; then
                echo ${experiment_id}
                sbatch \
                    --gres=gpu:1 \
                    -J ${experiment_id} \
                    -o ${persistent_dir}/logs/%x.%j.out \
                    -e ${persistent_dir}/logs/%x.%j.err \
                    python_job.sh experiments/seat_debias.py \
                        --tests ${seat_tests} \
                        --model ${model} \
                        --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                        --projection_matrix "${persistent_dir}/results/projection_matrix/projection_m-${debiased_model_to_base_model[${model}]}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}_s-0_p-${pre_assignment_type}.pt" \
                        --bias_type ${bias_type} \
                        --pre_assignment_type ${pre_assignment_type}
            fi
        done
    done
done

for model in ${sal_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        for pre_assignment_type in ${pre_assignment_types[@]}; do
            experiment_id="seat_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}_p-${pre_assignment_type}"
            if [ ! -f "${persistent_dir}/results/seat/${experiment_id}.json" ]; then
                echo ${experiment_id}
                sbatch \
                    --gres=gpu:1 \
                    -J ${experiment_id} \
                    -o ${persistent_dir}/logs/%x.%j.out \
                    -e ${persistent_dir}/logs/%x.%j.err \
                    python_job.sh experiments/seat_debias.py \
                        --tests ${seat_tests} \
                        --model ${model} \
                        --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                        --projection_matrix "${persistent_dir}/results/eigenvectors/projection_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}_s-0_p-${pre_assignment_type}.pt" \
                        --bias_type ${bias_type} \
                        --pre_assignment_type ${pre_assignment_type}
            fi
        done
    done
done
