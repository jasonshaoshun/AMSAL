#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

for model in ${masked_lm_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="crows_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
            echo ${experiment_id}
            sbatch \
                --gres=gpu:1 \
                -J ${experiment_id} \
                -o ${persistent_dir}/logs/%x.%j.out \
                -e ${persistent_dir}/logs/%x.%j.err \
                python_job.sh experiments/crows.py \
                    --model ${model} \
                    --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                    --bias_type ${bias_type}
        fi
    done
done


for model in ${causal_lm_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="crows_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
            echo ${experiment_id}
            sbatch \
                --gres=gpu:1 \
                -J ${experiment_id} \
                -o ${persistent_dir}/logs/%x.%j.out \
                -e ${persistent_dir}/logs/%x.%j.err \
                python_job.sh experiments/crows.py \
                    --model ${model} \
                    --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                    --bias_type ${bias_type}
        fi
    done
done







# ##################################################################

# Just use in the case when the GPU node failed to skip examples in the code

# ##################################################################


# export PYTHONPATH="${PYTHONPATH}:/bask/projects/j/jlxi8926-auto-sum/shun/code/unsup_svd/src/bias-bench"
# export TRANSFORMERS_CACHE=/bask/projects/j/jlxi8926-auto-sum/shun/code/unsup_svd/src/bias-bench/cache
# export HF_DATASETS_CACHE=/bask/projects/j/jlxi8926-auto-sum/shun/code/unsup_svd/src/bias-bench/cache

# for model in ${masked_lm_models[@]}; do
#     for bias_type in ${bias_types[@]}; do
#         experiment_id="crows_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
#         if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
#             echo ${experiment_id}
#             python experiments/crows.py \
#                 --model ${model} \
#                 --model_name_or_path ${model_to_model_name_or_path[${model}]} \
#                 --bias_type ${bias_type} \
#                 --persistent_dir ${persistent_dir}
#         fi
#     done
# done


# for model in ${causal_lm_models[@]}; do
#     for bias_type in ${bias_types[@]}; do
#         experiment_id="crows_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
#         if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
#             echo ${experiment_id}
#             python experiments/crows.py \
#                 --model ${model} \
#                 --model_name_or_path ${model_to_model_name_or_path[${model}]} \
#                 --bias_type ${bias_type} \
#                 --persistent_dir ${persistent_dir}
#         fi
#     done
# done
