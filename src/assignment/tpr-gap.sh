persistent_dir="/bask/projects/j/jlxi8926-auto-sum/shun/code/USAL/"

Debiasing_method=(
    # "SAL"
    "INLP"
)

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
# AM_Ratios=(
#     "0.8"
# )
# bias="race"
# eigenvectors_to_remove=2

# for debiasing_method in ${Debiasing_method[@]}; do
#     for assignment in ${Assignment_option[@]}; do
#         for assignment_ratio in ${AM_Ratios[@]}; do
#             experiment_id="tpr-gap_e-${experiment}_m-${model}_v-${variable_with_changing_ratio}_r-${assignment_ratio}_d-${debiasing_method}_a-${assignment}"
#             echo ${experiment_id}
#             sbatch \
#                 --gres=gpu:1 \
#                 -J ${experiment_id} \
#                 -o ${persistent_dir}/src/assignment/logs/${experiment}/${variable_with_changing_ratio}/${assignment_ratio}/tpr/%x.%j.out \
#                 -e ${persistent_dir}/src/assignment/logs/${experiment}/${variable_with_changing_ratio}/${assignment_ratio}/tpr/%x.%j.err \
#                 src/assignment/python_job.sh src/assignment/tpr-gap.py \
#                     --persistent_dir ${persistent_dir} \
#                     --debiasing ${debiasing_method} \
#                     --model ${model} \
#                     --bias ${bias} \
#                     --experiment ${experiment}\
#                     --assignment ${assignment} \
#                     --projection_matrix_file "${persistent_dir}/data/projection_matrix/${experiment}/${variable_with_changing_ratio}/${assignment_ratio}/${debiasing_method}_${bias}_${assignment}" \
#                     --projection_matrix_path "${persistent_dir}/data/projection_matrix/${experiment}/${variable_with_changing_ratio}/${assignment_ratio}" \
#                     --dataset_path "${persistent_dir}/data/${experiment}/${variable_with_changing_ratio}/" \
#                     --save_path "${persistent_dir}/src/assignment/results/${experiment}/${variable_with_changing_ratio}/" \
#                     --assignment_ratio ${assignment_ratio} \
#                     --train_ratio ${assignment_ratio} \
#                     --test_ratio ${assignment_ratio} \
#                     --eigenvectors_to_remove ${eigenvectors_to_remove}
#         done
#     done
# done



# # Run code.
# experiment="deepmoji"
# model="deepmoji"
# variable_with_changing_ratio="ratio_on_race"
# AM_Ratios=(
#     "0.5"
#     "0.8"
# )
# bias="race"
# eigenvectors_to_remove=2

# for debiasing_method in ${Debiasing_method[@]}; do
#     for assignment in ${Assignment_option[@]}; do
#         for assignment_ratio in ${AM_Ratios[@]}; do
#             experiment_id="tpr-gap_e-${experiment}_m-${model}_v-${variable_with_changing_ratio}_r-${assignment_ratio}_d-${debiasing_method}_a-${assignment}"
#             echo ${experiment_id}
#             sbatch \
#                 --gres=gpu:1 \
#                 -J ${experiment_id} \
#                 -o ${persistent_dir}/src/assignment/logs/${experiment}/${variable_with_changing_ratio}/${assignment_ratio}/tpr/%x.%j.out \
#                 -e ${persistent_dir}/src/assignment/logs/${experiment}/${variable_with_changing_ratio}/${assignment_ratio}/tpr/%x.%j.err \
#                 src/assignment/python_job.sh src/assignment/tpr-gap.py \
#                     --persistent_dir ${persistent_dir} \
#                     --debiasing ${debiasing_method} \
#                     --model ${model} \
#                     --bias ${bias} \
#                     --experiment ${experiment}\
#                     --assignment ${assignment} \
#                     --projection_matrix_file "${persistent_dir}/data/projection_matrix/${experiment}/${variable_with_changing_ratio}/${assignment_ratio}/${debiasing_method}_${bias}_${assignment}" \
#                     --projection_matrix_path "${persistent_dir}/data/projection_matrix/${experiment}/${variable_with_changing_ratio}/${assignment_ratio}" \
#                     --dataset_path "${persistent_dir}/data/${experiment}/${variable_with_changing_ratio}/" \
#                     --save_path "${persistent_dir}/src/assignment/results/${experiment}/${variable_with_changing_ratio}/" \
#                     --assignment_ratio ${assignment_ratio} \
#                     --train_ratio ${assignment_ratio} \
#                     --test_ratio ${assignment_ratio} \
#                     --eigenvectors_to_remove ${eigenvectors_to_remove}
#         done
#     done
# done




















Models=(
    "BertModel"
    # "FastText"
)
bias="gender"
experiment="biography"
eigenvectors_to_remove=2

for model in ${Models[@]}; do
    for debiasing_method in ${Debiasing_method[@]}; do
        for assignment in ${Assignment_option[@]}; do
            experiment_id="tpr-gap_e-${experiment}_m-${model}_d-${debiasing_method}_a-${assignment}"
            echo ${experiment_id}
            sbatch \
                --gres=gpu:1 \
                -J ${experiment_id} \
                -o ${persistent_dir}/src/assignment/logs/${experiment}/${model}/tpr/%x.%j.out \
                -e ${persistent_dir}/src/assignment/logs/${experiment}/${model}/tpr/%x.%j.err \
                src/assignment/python_job.sh src/assignment/tpr-gap.py \
                    --persistent_dir ${persistent_dir} \
                    --debiasing ${debiasing_method} \
                    --model ${model} \
                    --bias ${bias} \
                    --experiment ${experiment} \
                    --assignment ${assignment} \
                    --projection_matrix_file "${persistent_dir}/data/projection_matrix/${experiment}/${model}/${debiasing_method}_${bias}_${assignment}" \
                    --projection_matrix_path "${persistent_dir}/data/projection_matrix/${experiment}/${model}" \
                    --dataset_path "${persistent_dir}/data/${experiment}/${model}/all.npz" \
                    --save_path "${persistent_dir}/src/assignment/results/${experiment}/${model}/" \
                    --eigenvectors_to_remove ${eigenvectors_to_remove}
        done
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
eigenvectors_to_remove=2


for model in ${Models[@]}; do
    for debiasing_method in ${Debiasing_method[@]}; do
        for seed in ${Seeds[@]}; do
            experiment_id="tpr-gap_e-${experiment}_m-${model}_d-${debiasing_method}_a-${assignment}_s-${seed}"
            echo ${experiment_id}
            sbatch \
                --gres=gpu:1 \
                -J ${experiment_id} \
                -o ${persistent_dir}/src/assignment/logs/${experiment}/${model}/tpr/%x.%j.out \
                -e ${persistent_dir}/src/assignment/logs/${experiment}/${model}/tpr/%x.%j.err \
                src/assignment/python_job.sh src/assignment/tpr-gap.py \
                    --persistent_dir ${persistent_dir} \
                    --debiasing ${debiasing_method} \
                    --model ${model} \
                    --bias ${bias} \
                    --seed ${seed} \
                    --experiment ${experiment} \
                    --assignment ${assignment} \
                    --projection_matrix_file "${persistent_dir}/data/projection_matrix/${experiment}/${model}/${debiasing_method}_${bias}_${assignment}_seed-${seed}" \
                    --projection_matrix_path "${persistent_dir}/data/projection_matrix/${experiment}/${model}" \
                    --dataset_path "${persistent_dir}/data/${experiment}/${model}/all.npz" \
                    --save_path "${persistent_dir}/src/assignment/results/${experiment}/${model}/" \
                    --eigenvectors_to_remove ${eigenvectors_to_remove}
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

# experiment="biography-different-partial-n"
# Models=(
#     "BertModel"
#     "FastText"
# )
# bias="gender"
# assignment="partialSup"
# eigenvectors_to_remove=2

# for model in ${Models[@]}; do
#     for debiasing_method in ${Debiasing_method[@]}; do
#         for supervision_ratio in ${Partial_N[@]}; do
#             experiment_id="tpr-gap_e-${experiment}_m-${model}_d-${debiasing_method}_a-${assignment}"
#             echo ${experiment_id}
#             sbatch \
#                 --gres=gpu:1 \
#                 -J ${experiment_id} \
#                 -o ${persistent_dir}/src/assignment/logs/${experiment}/${model}/tpr/%x.%j.out \
#                 -e ${persistent_dir}/src/assignment/logs/${experiment}/${model}/tpr/%x.%j.err \
#                 src/assignment/python_job.sh src/assignment/tpr-gap.py \
#                     --persistent_dir ${persistent_dir} \
#                     --debiasing ${debiasing_method} \
#                     --model ${model} \
#                     --bias ${bias} \
#                     --supervision_ratio ${supervision_ratio} \
#                     --experiment ${experiment} \
#                     --assignment ${assignment} \
#                     --projection_matrix_file "${persistent_dir}/data/projection_matrix/${experiment}/${model}/${debiasing_method}_${bias}_${assignment}_np-${supervision_ratio}" \
#                     --projection_matrix_path "${persistent_dir}/data/projection_matrix/${experiment}/${model}" \
#                     --dataset_path "${persistent_dir}/data/${experiment}/${model}/all.npz" \
#                     --save_path "${persistent_dir}/src/assignment/results/${experiment}/${model}/" \
#                     --eigenvectors_to_remove ${eigenvectors_to_remove}
#         done
#     done
# done



