persistent_dir="/bask/projects/j/jlxi8926-auto-sum/shun/code/USAL/"

Assignment_option=(
    "Kmeans"
    "Oracle"
    "Sal"
    "partialSup"
)


# model="BertModel"
model="FastText"



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










# # Run code.

# bias="age-gender"
# Experiments_list=(
#     # "twitter-concat-sentiment"
#     # "twitter-short-sentiment"
#     "twitter-concat-sentiment-v4"
# )

# for experiment in ${Experiments_list[@]}; do
#     for assignment in ${Assignment_option[@]}; do
#         experiment_id="compute-inlp-projection_e-${experiment}_m-${model}_a-${assignment}"
#         echo ${experiment_id}
#         echo ${experiment_to_data_version[${experiment}]}
#         sbatch \
#             --gres=gpu:1 \
#             -J ${experiment_id} \
#             -o ${persistent_dir}/src/assignment/logs/${experiment}/inlp/%x.%j.out \
#             -e ${persistent_dir}/src/assignment/logs/${experiment}/inlp/%x.%j.err \
#             src/assignment/python_job.sh src/assignment/inlp_projection_twitter.py \
#                 --persistent_dir ${persistent_dir} \
#                 --model ${model} \
#                 --assignment ${assignment} \
#                 --bias ${bias} \
#                 --assignment_path "${persistent_dir}/data/projection_matrix/${experiment}/${model}/SAL_${bias}_${assignment}.mat" \
#                 --dataset_path "${persistent_dir}/data/${experiment}/${model}/${bias}.npz" \
#                 --save_path "${persistent_dir}/data/projection_matrix/${experiment}/${model}" \
#                 --dataset_version ${experiment_to_data_version[${experiment}]}
#     done
# done

















# Run code.

Partial_N=(
    "0"
    "0.1"
    "0.2"
    "0.3"
    "0.4"
    "0.5"
    "0.6"
    "0.7"
    "0.8"
    "0.9"
    "1"
)


bias="age-gender"
Experiments_list=(
    # "twitter-different-partial-n-concat"
    # "twitter-different-partial-n-short"
    "twitter-different-partial-n-concat-v4"
)
assignment="partialSup"
for experiment in ${Experiments_list[@]}; do
    for supervision_ratio in ${Partial_N[@]}; do
        experiment_id="compute-inlp-projection_e-${experiment}_m-${model}"
        echo ${experiment_id}
        sbatch \
            --gres=gpu:1 \
            -J ${experiment_id} \
            -o ${persistent_dir}/src/assignment/logs/${experiment}/inlp/%x.%j.out \
            -e ${persistent_dir}/src/assignment/logs/${experiment}/inlp/%x.%j.err \
            src/assignment/python_job.sh src/assignment/inlp_projection_twitter.py \
                --persistent_dir ${persistent_dir} \
                --model ${model} \
                --assignment ${assignment} \
                --bias ${bias} \
                --supervision_ratio ${supervision_ratio} \
                --assignment_path "${persistent_dir}/data/projection_matrix/${experiment}/${model}/SAL_${bias}_${assignment}_np-${supervision_ratio}.mat" \
                --dataset_path "${persistent_dir}/data/${experiment}/${model}/${bias}.npz" \
                --save_path "${persistent_dir}/data/projection_matrix/${experiment}/${model}" \
                --dataset_version ${experiment_to_data_version[${experiment}]}

    done
done














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


# bias="age-gender"
# Experiments_list=(
#     # "twitter-labelled-by-overlap-concat"
#     # "twitter-labelled-by-overlap-short"
#     "twitter-labelled-by-overlap-concat-v4"
# )

# assignment="partialSup"
# # Run code.
# for experiment in ${Experiments_list[@]}; do
#     for seed in ${Seeds[@]}; do
#         experiment_id="compute-inlp-projection_e-${experiment}_m-${model}_s-${seed}"
#         echo ${experiment_id}
#         sbatch \
#             --gres=gpu:1 \
#             -J ${experiment_id} \
#             -o ${persistent_dir}/src/assignment/logs/${experiment}/inlp/%x.%j.out \
#             -e ${persistent_dir}/src/assignment/logs/${experiment}/inlp/%x.%j.err \
#             src/assignment/python_job.sh src/assignment/inlp_projection_twitter.py \
#                 --persistent_dir ${persistent_dir} \
#                 --model ${model} \
#                 --assignment ${assignment} \
#                 --bias ${bias} \
#                 --seed ${seed} \
#                 --assignment_path "${persistent_dir}/data/projection_matrix/${experiment}/${model}/SAL_${bias}_${assignment}_seed-${seed}.mat" \
#                 --dataset_path "${persistent_dir}/data/${experiment}/${model}/${bias}.npz" \
#                 --save_path "${persistent_dir}/data/projection_matrix/${experiment}/${model}" \
#                 --dataset_version ${experiment_to_data_version[${experiment}]}
#     done
# done



