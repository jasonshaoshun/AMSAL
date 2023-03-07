#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

        

for model in ${crows_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="export-crows_m-${model}_t-${bias_type}"
        echo ${experiment_id}
        sbatch \
            --gres=gpu:1 \
            -o ${persistent_dir}/%x.%j.out \
            -e ${persistent_dir}/%x.%j.err \
            python_job.sh export/crows.py \
                --model_type ${model} \
                --bias_type ${bias_type}
    done
done

# for model in ${inlp_masked_lm_models[@]}; do
#     for bias_type in ${bias_types[@]}; do
#         for pre_assignment_type in ${pre_assignment_types[@]}; do
#             experiment_id="crows_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}_p-${pre_assignment_type}"
#             if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
#                 echo ${experiment_id}
#                 sbatch \
#                     --gres=gpu:1 \
#                     --time ${time[${model_to_model_name_or_path[${model}]}]} \
#                     -J ${experiment_id} \
#                     -o ${persistent_dir}/logs/%x.%j.out \
#                     -e ${persistent_dir}/logs/%x.%j.err \
#                     python_job.sh experiments/crows_debias.py \
#                         --model ${model} \
#                         --model_name_or_path ${model_to_model_name_or_path[${model}]} \
#                         --projection_matrix "${persistent_dir}/results/projection_matrix/projection_m-${debiased_model_to_base_model[${model}]}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}_s-0_p-${pre_assignment_type}.pt" \
#                         --bias_type ${bias_type} \
#                         --pre_assignment_type ${pre_assignment_type}
#             fi
#         done
#     done
# done


# for model in ${inlp_causal_lm_models[@]}; do
#     for bias_type in ${bias_types[@]}; do
#         for pre_assignment_type in ${pre_assignment_types[@]}; do
#             experiment_id="crows_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}_p-${pre_assignment_type}"
#             if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
#                 echo ${experiment_id}
#                 sbatch \
#                     --gres=gpu:1 \
#                     --time ${time[${model_to_model_name_or_path[${model}]}]} \
#                     -J ${experiment_id} \
#                     -o ${persistent_dir}/logs/%x.%j.out \
#                     -e ${persistent_dir}/logs/%x.%j.err \
#                     python_job.sh experiments/crows_debias.py \
#                         --model ${model} \
#                         --model_name_or_path ${model_to_model_name_or_path[${model}]} \
#                         --projection_matrix "${persistent_dir}/results/projection_matrix/projection_m-${debiased_model_to_base_model[${model}]}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}_s-0_p-${pre_assignment_type}.pt" \
#                         --bias_type ${bias_type} \
#                         --pre_assignment_type ${pre_assignment_type}
#             fi
#         done
#     done
# done