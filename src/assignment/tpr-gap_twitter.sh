persistent_dir="/bask/projects/j/jlxi8926-auto-sum/shun/code/USAL/"

Debiasing_method=(
    "SAL"
    "INLP"
)


Assignment_option=(
    "Kmeans"
    "Oracle"
    "Sal"
    "partialSup"
)

model="BertModel"
# model="FastText"


declare -A  experiment_to_data_version=(
    ["twitter-different-partial-n-concat"]="v3"
    ["twitter-different-partial-n-short"]="v3"
    ["twitter-concat-sentiment"]="v3"
    ["twitter-short-sentiment"]="v3"
    ["twitter-labelled-by-overlap-concat"]="v3"
    ["twitter-labelled-by-overlap-short"]="v3"
    ["twitter-different-partial-n-concat-v4"]="v4"
    ["twitter-concat-sentiment-v4"]="v4"
    ["twitter-labelled-by-overlap-concat-v4"]="v4"
)




# Run code.

bias="age-gender"
num_eigenvectors_to_remove=6
Experiments_list=(
    # "twitter-concat-sentiment"
    # "twitter-short-sentiment"
    "twitter-concat-sentiment-v4"
)

for experiment in ${Experiments_list[@]}; do
    for debiasing_method in ${Debiasing_method[@]}; do
        for assignment in ${Assignment_option[@]}; do
            experiment_id="tpr-gap_e-${experiment}_m-${model}_d-${debiasing_method}_a-${assignment}"
            echo ${experiment_id}
            sbatch \
                --gres=gpu:1 \
                -J ${experiment_id} \
                -o ${persistent_dir}/src/assignment/logs/${experiment}/tpr/%x.%j.out \
                -e ${persistent_dir}/src/assignment/logs/${experiment}/tpr/%x.%j.err \
                src/assignment/python_job.sh src/assignment/tpr-gap_twitter.py \
                    --persistent_dir ${persistent_dir} \
                    --debiasing ${debiasing_method} \
                    --bias ${bias} \
                    --model ${model} \
                    --experiment ${experiment} \
                    --assignment ${assignment} \
                    --projection_matrix_file "${persistent_dir}/data/projection_matrix/${experiment}/${model}/${debiasing_method}_${bias}_${assignment}" \
                    --projection_matrix_path "${persistent_dir}/data/projection_matrix/${experiment}/${model}/" \
                    --dataset_path "${persistent_dir}/data/${experiment}/${model}/${bias}.npz" \
                    --save_path "${persistent_dir}/src/assignment/results/${experiment}/" \
                    --num_eigenvectors_to_remove ${num_eigenvectors_to_remove} \
                    --dataset_version ${experiment_to_data_version[${experiment}]}
        done
    done
done













# # Run code.

# Partial_N=(
#     "0"
#     "0.1"
#     "0.2"
#     "0.3"
#     "0.4"
#     "0.5"
#     "0.6"
#     "0.7"
#     "0.8"
#     "0.9"
#     "1"
# )

# bias="age-gender"
# num_eigenvectors_to_remove=6
# Experiments_list=(
#     # "twitter-different-partial-n-concat"
#     # "twitter-different-partial-n-short"
#     "twitter-different-partial-n-concat-v4"
# )
# assignment="partialSup"

# for experiment in ${Experiments_list[@]}; do
#     for debiasing_method in ${Debiasing_method[@]}; do
#         for supervision_ratio in ${Partial_N[@]}; do
#             experiment_id="tpr-gap_e-${experiment}_m-${model}_d-${debiasing_method}_a-${assignment}_r-${supervision_ratio}"
#             echo ${experiment_id}
#             sbatch \
#                 --gres=gpu:1 \
#                 -J ${experiment_id} \
#                 -o ${persistent_dir}/src/assignment/logs/${experiment}/tpr/%x.%j.out \
#                 -e ${persistent_dir}/src/assignment/logs/${experiment}/tpr/%x.%j.err \
#                 src/assignment/python_job.sh src/assignment/tpr-gap_twitter.py \
#                     --persistent_dir ${persistent_dir} \
#                     --debiasing ${debiasing_method} \
#                     --bias ${bias} \
#                     --model ${model} \
#                     --experiment ${experiment} \
#                     --assignment ${assignment} \
#                     --supervision_ratio ${supervision_ratio} \
#                     --projection_matrix_path "${persistent_dir}/data/projection_matrix/${experiment}/${model}/"\
#                     --projection_matrix_file "${persistent_dir}/data/projection_matrix/${experiment}/${model}/${debiasing_method}_${bias}_${assignment}_np-${supervision_ratio}" \
#                     --dataset_path "${persistent_dir}/data/${experiment}/${model}/${bias}.npz" \
#                     --save_path "${persistent_dir}/src/assignment/results/${experiment}/" \
#                     --num_eigenvectors_to_remove ${num_eigenvectors_to_remove} \
#                     --dataset_version ${experiment_to_data_version[${experiment}]}
#         done
#     done
# done















# # Run code.
# bias="age-gender"
# assignment="partialSup"
# num_eigenvectors_to_remove=6
# Experiments_list=(
#     # "twitter-labelled-by-overlap-concat"
#     # "twitter-labelled-by-overlap-short"
#     "twitter-labelled-by-overlap-concat-v4"
# )
# Seeds=(
#     "0"
#     "1"
#     "2"
#     "3"
#     "4"
#     "5"
#     "6"
#     "7"
#     "8"
#     "9"
# )


# # Run code.
# for experiment in ${Experiments_list[@]}; do
#     for seed in ${Seeds[@]}; do
#         for debiasing_method in ${Debiasing_method[@]}; do
#             experiment_id="tpr-gap_e-${experiment}_m-${model}_d-${debiasing_method}_a-${assignment}"
#             echo ${experiment_id}
#             sbatch \
#                 --gres=gpu:1 \
#                 -J ${experiment_id} \
#                 -o ${persistent_dir}/src/assignment/logs/${experiment}/tpr/%x.%j.out \
#                 -e ${persistent_dir}/src/assignment/logs/${experiment}/tpr/%x.%j.err \
#                 src/assignment/python_job.sh src/assignment/tpr-gap_twitter.py \
#                     --persistent_dir ${persistent_dir} \
#                     --debiasing ${debiasing_method} \
#                     --bias ${bias} \
#                     --model ${model} \
#                     --experiment ${experiment} \
#                     --assignment ${assignment} \
#                     --projection_matrix_file "${persistent_dir}/data/projection_matrix/${experiment}/${model}/${debiasing_method}_${bias}_${assignment}_seed-${seed}" \
#                     --projection_matrix_path "${persistent_dir}/data/projection_matrix/${experiment}/${model}" \
#                     --dataset_path "${persistent_dir}/data/${experiment}/${model}/${bias}.npz" \
#                     --save_path "${persistent_dir}/src/assignment/results/${experiment}/" \
#                     --seed ${seed}\
#                     --num_eigenvectors_to_remove ${num_eigenvectors_to_remove} \
#                     --dataset_version ${experiment_to_data_version[${experiment}]}
#         done
#     done
# done

