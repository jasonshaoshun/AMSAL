persistent_dir="/bask/projects/j/jlxi8926-auto-sum/shun/code/unsup_svd/"



Assignment_option=(
    # "Kmeans"
    "Oracle"
    # "Sal"
    # "partialSup"
)









# Run code.
experiment="deepmoji"
model="deepmoji"
variable_with_changing_ratio="ratio_on_race"
Ratios=(
    "0.5"
    # "0.8"
)
bias="race"

for dataset_ratio in ${Ratios[@]}; do
    for assignment in ${Assignment_option[@]}; do
        experiment_id="best-hyperparameter_e-${experiment}_m-${model}_r-${variable_with_changing_ratio}_a-${assignment}"
        echo ${experiment_id}
        sbatch \
            --gres=gpu:1 \
            -J ${experiment_id} \
            -o ${persistent_dir}/src/assignment/logs/${experiment}/${variable_with_changing_ratio}/${dataset_ratio}/inlp/%x.%j.out \
            -e ${persistent_dir}/src/assignment/logs/${experiment}/${variable_with_changing_ratio}/${dataset_ratio}/inlp/%x.%j.err \
            src/assignment/python_job.sh src/assignment/inlp_hyperparameter.py \
                --assignment_path "data/projection_matrix/${experiment}/${variable_with_changing_ratio}/${dataset_ratio}/SAL_${bias}_${assignment}.mat" \
                --dataset_path "data/${experiment}/${variable_with_changing_ratio}/${dataset_ratio}/all.npz" \
                --save_path "data/projection_matrix/${experiment}/${variable_with_changing_ratio}/${dataset_ratio}/"
    done
done





experiment="biography"
Models=(
    "BertModel"
    "FastText"
)
bias="gender"

for model in ${Models[@]}; do
    for assignment in ${Assignment_option[@]}; do
        experiment_id="compute-inlp-projection_e-${experiment}_m-${model}_a-${assignment}"
        echo ${experiment_id}
        sbatch \
            --gres=gpu:1 \
            -J ${experiment_id} \
            -o ${persistent_dir}/src/assignment/logs/${experiment}/${model}/%x.%j.out \
            -e ${persistent_dir}/src/assignment/logs/${experiment}/${model}/%x.%j.err \
            src/assignment/python_job.sh src/assignment/inlp_hyperparameter.py \
                --assignment_path "data/projection_matrix/${experiment}/${model}/SAL_${bias}_${assignment}.mat" \
                --save_path "data/projection_matrix/${experiment}/${model}/" \
                --dataset_path "data/${experiment}/${model}/all.npz"
    done
done
