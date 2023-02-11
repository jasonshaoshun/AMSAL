persistent_dir="/bask/projects/j/jlxi8926-auto-sum/shun/code/USAL/"



Assignment_option=(
    # "Kmeans"
    # "Oracle"
    "Sal"
    # "partialSup"
)



# # Run code.
# experiment="deepmoji"
# model="deepmoji"
# variable_with_changing_ratio="ratio_on_sent"
# Ratios=(
#     "0.8"
# )
# bias="race"
# for dataset_ratio in ${Ratios[@]}; do
#     for assignment in ${Assignment_option[@]}; do
#         experiment_id="compute-inlp-projection_e-${experiment}_m-${model}_r-${variable_with_changing_ratio}_a-${assignment}"
#         echo ${experiment_id}
#         sbatch \
#             --gres=gpu:1 \
#             -J ${experiment_id} \
#             -o ${persistent_dir}/src/assignment/logs/${experiment}/${variable_with_changing_ratio}/${dataset_ratio}/inlp/%x.%j.out \
#             -e ${persistent_dir}/src/assignment/logs/${experiment}/${variable_with_changing_ratio}/${dataset_ratio}/inlp/%x.%j.err \
#             src/assignment/python_job.sh src/assignment/inlp_projection.py \
#                 --persistent_dir ${persistent_dir} \
#                 --model ${model} \
#                 --assignment ${assignment} \
#                 --assignment_path "data/projection_matrix/${experiment}/${variable_with_changing_ratio}/${dataset_ratio}/SAL_${bias}_${assignment}.mat" \
#                 --dataset_path "data/${experiment}/${variable_with_changing_ratio}/${dataset_ratio}/all.npz" \
#                 --save_path "data/projection_matrix/${experiment}/${variable_with_changing_ratio}/${dataset_ratio}/" \
#                 --bias ${bias} \
#                 --experiment ${experiment}
#     done
# done

# # Run code.
# experiment="deepmoji"
# model="deepmoji"
# variable_with_changing_ratio="ratio_on_race"
# Ratios=(
#     "0.5"
#     "0.8"
# )
# bias="race"

# for dataset_ratio in ${Ratios[@]}; do
#     for assignment in ${Assignment_option[@]}; do
#         experiment_id="compute-inlp-projection_e-${experiment}_m-${model}_r-${variable_with_changing_ratio}_a-${assignment}"
#         echo ${experiment_id}
#         sbatch \
#             --gres=gpu:1 \
#             -J ${experiment_id} \
#             -o ${persistent_dir}/src/assignment/logs/${experiment}/${variable_with_changing_ratio}/${dataset_ratio}/inlp/%x.%j.out \
#             -e ${persistent_dir}/src/assignment/logs/${experiment}/${variable_with_changing_ratio}/${dataset_ratio}/inlp/%x.%j.err \
#             src/assignment/python_job.sh src/assignment/inlp_projection.py \
#                 --persistent_dir ${persistent_dir} \
#                 --model ${model} \
#                 --assignment ${assignment} \
#                 --assignment_path "data/projection_matrix/${experiment}/${variable_with_changing_ratio}/${dataset_ratio}/SAL_${bias}_${assignment}.mat" \
#                 --dataset_path "data/${experiment}/${variable_with_changing_ratio}/${dataset_ratio}/all.npz" \
#                 --save_path "data/projection_matrix/${experiment}/${variable_with_changing_ratio}/${dataset_ratio}/" \
#                 --bias ${bias} \
#                 --experiment ${experiment}
#     done
# done




experiment="biography"
Models=(
    "BertModel"
    # "FastText"
)
bias="gender"

for model in ${Models[@]}; do
    for assignment in ${Assignment_option[@]}; do
        experiment_id="compute-inlp-projection_e-${experiment}_m-${model}_a-${assignment}"
        echo ${experiment_id}
        sbatch \
            --gres=gpu:1 \
            -J ${experiment_id} \
            -o ${persistent_dir}/src/assignment/logs/${experiment}/${model}/inlp/%x.%j.out \
            -e ${persistent_dir}/src/assignment/logs/${experiment}/${model}/inlp/%x.%j.err \
            src/assignment/python_job.sh src/assignment/inlp_projection.py \
                --persistent_dir ${persistent_dir} \
                --model ${model} \
                --bias ${bias} \
                --experiment ${experiment} \
                --assignment ${assignment} \
                --assignment_path "data/projection_matrix/${experiment}/${model}/SAL_${bias}_${assignment}.mat" \
                --save_path "data/projection_matrix/${experiment}/${model}/" \
                --dataset_path "data/${experiment}/${model}/all.npz"
    done
done



Seeds=(
    "0"
    # "1"
    # "2"
    # "3"
    # "4"
    # "5"
    # "6"
    # "7"
    # "8"
    # "9"
)

experiment="biography-labelled-by-overlap"
Models=(
    "BertModel"
    # "FastText"
)
bias="gender"
assignment="partialSup"

for model in ${Models[@]}; do
    for seed in ${Seeds[@]}; do
        experiment_id="compute-inlp-projection_e-${experiment}_m-${model}_a-${assignment}_s-${seed}"
        echo ${experiment_id}
        sbatch \
            --gres=gpu:1 \
            -J ${experiment_id} \
            -o ${persistent_dir}/src/assignment/logs/${experiment}/${model}/inlp/%x.%j.out \
            -e ${persistent_dir}/src/assignment/logs/${experiment}/${model}/inlp/%x.%j.err \
            src/assignment/python_job.sh src/assignment/inlp_projection.py \
                --persistent_dir ${persistent_dir} \
                --model ${model} \
                --bias ${bias} \
                --experiment ${experiment} \
                --assignment ${assignment} \
                --assignment_path "data/projection_matrix/${experiment}/${model}/SAL_${bias}_${assignment}_seed-${seed}.mat" \
                --save_path "data/projection_matrix/${experiment}/${model}/" \
                --dataset_path "data/${experiment}/${model}/all.npz" \
                --seed ${seed}
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

# experiment="biography-different-partial-n"
# Models=(
#     "BertModel"
#     "FastText"
# )
# bias="gender"
# assignment="partialSup"

# for model in ${Models[@]}; do
#     for supervision_ratio in ${Partial_N[@]}; do
#         experiment_id="compute-inlp-projection_e-${experiment}_m-${model}_a-${assignment}"
#         echo ${experiment_id}
#         sbatch \
#             --gres=gpu:1 \
#             -J ${experiment_id} \
#             -o ${persistent_dir}/src/assignment/logs/${experiment}/${model}/inlp/%x.%j.out \
#             -e ${persistent_dir}/src/assignment/logs/${experiment}/${model}/inlp/%x.%j.err \
#             src/assignment/python_job.sh src/assignment/inlp_projection.py \
#                 --persistent_dir ${persistent_dir} \
#                 --model ${model} \
#                 --bias ${bias} \
#                 --supervision_ratio ${supervision_ratio} \
#                 --experiment ${experiment} \
#                 --assignment ${assignment} \
#                 --assignment_path "data/projection_matrix/${experiment}/${model}/SAL_${bias}_${assignment}_np-${supervision_ratio}.mat" \
#                 --save_path "data/projection_matrix/${experiment}/${model}/" \
#                 --dataset_path "data/${experiment}/${model}/all.npz" 
#     done
# done


