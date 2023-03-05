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





# Run code.
experiment="deepmoji"
model="deepmoji"
variable_with_changing_ratio="ratio_on_sent"
AM_Ratios=(
    "0.8"
)
bias="race"
eigenvectors_to_remove=2

for debiasing_method in ${Debiasing_method[@]}; do
    for assignment in ${Assignment_option[@]}; do
        for assignment_ratio in ${AM_Ratios[@]}; do
            experiment_id="tpr-gap_e-${experiment}_m-${model}_v-${variable_with_changing_ratio}_r-${assignment_ratio}_d-${debiasing_method}_a-${assignment}"
            echo ${experiment_id}
            sbatch \
                --gres=gpu:1 \
                -J ${experiment_id} \
                -o ${persistent_dir}/src/assignment/logs/${experiment}/${variable_with_changing_ratio}/${assignment_ratio}/tpr/%x.%j.out \
                -e ${persistent_dir}/src/assignment/logs/${experiment}/${variable_with_changing_ratio}/${assignment_ratio}/tpr/%x.%j.err \
                src/assignment/python_job.sh src/assignment/tpr-gap.py \
                    --persistent_dir ${persistent_dir} \
                    --debiasing ${debiasing_method} \
                    --model ${model} \
                    --bias ${bias} \
                    --experiment ${experiment}\
                    --assignment ${assignment} \
                    --projection_matrix_file "${persistent_dir}/data/projection_matrix/${experiment}/${variable_with_changing_ratio}/${assignment_ratio}/${debiasing_method}_${bias}_${assignment}" \
                    --projection_matrix_path "${persistent_dir}/data/projection_matrix/${experiment}/${variable_with_changing_ratio}/${assignment_ratio}" \
                    --dataset_path "${persistent_dir}/data/${experiment}/${variable_with_changing_ratio}/" \
                    --save_path "${persistent_dir}/src/assignment/results/${experiment}/${variable_with_changing_ratio}/" \
                    --assignment_ratio ${assignment_ratio} \
                    --train_ratio ${assignment_ratio} \
                    --test_ratio ${assignment_ratio} \
                    --eigenvectors_to_remove ${eigenvectors_to_remove}
        done
    done
done



# Run code.
experiment="deepmoji"
model="deepmoji"
variable_with_changing_ratio="ratio_on_race"
AM_Ratios=(
    "0.5"
    "0.8"
)
bias="race"
eigenvectors_to_remove=2

for debiasing_method in ${Debiasing_method[@]}; do
    for assignment in ${Assignment_option[@]}; do
        for assignment_ratio in ${AM_Ratios[@]}; do
            experiment_id="tpr-gap_e-${experiment}_m-${model}_v-${variable_with_changing_ratio}_r-${assignment_ratio}_d-${debiasing_method}_a-${assignment}"
            echo ${experiment_id}
            sbatch \
                --gres=gpu:1 \
                -J ${experiment_id} \
                -o ${persistent_dir}/src/assignment/logs/${experiment}/${variable_with_changing_ratio}/${assignment_ratio}/tpr/%x.%j.out \
                -e ${persistent_dir}/src/assignment/logs/${experiment}/${variable_with_changing_ratio}/${assignment_ratio}/tpr/%x.%j.err \
                src/assignment/python_job.sh src/assignment/tpr-gap.py \
                    --persistent_dir ${persistent_dir} \
                    --debiasing ${debiasing_method} \
                    --model ${model} \
                    --bias ${bias} \
                    --experiment ${experiment}\
                    --assignment ${assignment} \
                    --projection_matrix_file "${persistent_dir}/data/projection_matrix/${experiment}/${variable_with_changing_ratio}/${assignment_ratio}/${debiasing_method}_${bias}_${assignment}" \
                    --projection_matrix_path "${persistent_dir}/data/projection_matrix/${experiment}/${variable_with_changing_ratio}/${assignment_ratio}" \
                    --dataset_path "${persistent_dir}/data/${experiment}/${variable_with_changing_ratio}/" \
                    --save_path "${persistent_dir}/src/assignment/results/${experiment}/${variable_with_changing_ratio}/" \
                    --assignment_ratio ${assignment_ratio} \
                    --train_ratio ${assignment_ratio} \
                    --test_ratio ${assignment_ratio} \
                    --eigenvectors_to_remove ${eigenvectors_to_remove}
        done
    done
done


