persistent_dir="/bask/projects/j/jlxi8926-auto-sum/shun/code/unsup_svd/"

Assignment_option=(
    # "Kmeans"
    "Oracle"
    # "Sal"
    # "partialSup"
)






model="BertModel"
# model="FastText"



# Run code.

bias="age-gender"
Experiments_list=(
    # "twitter-concat-sentiment"
    # "twitter-short-sentiment"
    "twitter-concat-sentiment-v4"
    # "twitter-short-sentiment-v4"
)

for experiment in ${Experiments_list[@]}; do
    for assignment in ${Assignment_option[@]}; do
        experiment_id="best-hyperparameter_e-${experiment}_m-${model}_a-${assignment}"
        echo ${experiment_id}
        sbatch \
            --gres=gpu:1 \
            -J ${experiment_id} \
            -o ${persistent_dir}/src/assignment/logs/${experiment}/%x.%j.out \
            -e ${persistent_dir}/src/assignment/logs/${experiment}/%x.%j.err \
            src/assignment/python_job.sh src/assignment/inlp_hyperparameter_twitter.py \
                --persistent_dir ${persistent_dir} \
                --model ${model} \
                --assignment_path "${persistent_dir}/data/projection_matrix/${experiment}/${model}/SAL_${bias}_${assignment}.mat" \
                --dataset_path "${persistent_dir}/data/${experiment}/${model}/${bias}.npz" \
                --save_path "${persistent_dir}/data/projection_matrix/${experiment}/${model}"
    done
done


