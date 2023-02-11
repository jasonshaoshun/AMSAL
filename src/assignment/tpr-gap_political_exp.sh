#!/bin/bash
#SBATCH --qos epsrc
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

module purge
module load baskerville
module load CUDA/11.1.1-GCC-10.2.0

# Load the required modules
eval "$(/bask/projects/j/jlxi8926-auto-sum/shun/anaconda3/bin/conda shell.bash hook)"
conda activate biasbench


export PYTHONPATH="${PYTHONPATH}:/bask/projects/j/jlxi8926-auto-sum/shun/code/unsup_svd/"

persistent_dir="/bask/projects/j/jlxi8926-auto-sum/shun/code/unsup_svd/"

model="BertModel"


Debiasing_method=(
    "SAL"
    "INLP"
)

Experiments=(
    # "concat_twitter_sentiment"
    # "concat_twitter_vulgar"
    # "normal_twitter_sentiment"
    # "normal_twitter_vulgar"
    "twitter-different-partial-n"
)

Biases=(
    "age-gender"
    # "age"
    # "gender"
)

Assignment_option=(
    # "Kmeans"
    # "Oracle"
    # "Sal"
    "partialSup"
)

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

bias_to_eigenvectors_removal=(
    ["age-gender"]=6
)

# Run code.
for debiasing_method in ${Debiasing_method[@]}; do
    for experiment in ${Experiments[@]}; do
        for bias in ${Biases[@]}; do
            for assignment in ${Assignment_option[@]}; do
                for supervision_ratio in ${Partial_N[@]}; do
                    experiment_id="tpr-gap_m-${model}_e-${experiment}_d-${debiasing_method}_a-${assignment}_s-${supervision_ratio}"
                    echo ${experiment_id}
                    sbatch \
                        --gres=gpu:1 \
                        -J ${experiment_id} \
                        -o ${persistent_dir}/src/assignment/logs/%x.%j.out \
                        -e ${persistent_dir}/src/assignment/logs/%x.%j.err \
                        src/assignment/python_job.sh src/assignment/tpr-gap_political_exp.py \
                            --persistent_dir ${persistent_dir} \
                            --debiasing ${debiasing_method} \
                            --model ${model} \
                            --experiment ${experiment} \
                            --assignment ${assignment} \
                            --bias ${bias} \
                            --supervision_ratio ${supervision_ratio} \
                            --dataset_path "${persistent_dir}/data/${experiment}/${model}/${bias}.npz" \
                            --eigenvector_path "${persistent_dir}/data/projection_matrix/${experiment}/${model}/${debiasing_method}_${bias}_${assignment}_np-${supervision_ratio}" \
                            --eigenvectors_to_keep ${bias_to_eigenvectors_removal[${bias}]} \
                            --file_contains_original_Z_and_assignment_Z "${persistent_dir}/data/projection_matrix/${experiment}/${model}/SAL_${bias}_${assignment}_np-${supervision_ratio}.mat"
                done
            done
        done
    done
done


    # dataset_to_dataset_folder = {
    #     "BertModel": f"/data/{args.experiment}/{args.model}/{args.model}_{args.bias}.npz",
    #     "Deepmoji": f"/data/{args.experiment}/{args.model}/{args.model}_{args.bias}.npz"
    # }

    # dataset_to_eigenvector_folder = {
    #     "BertModel": f"data/projection_matrix/{args.experiment}/{args.model}/{args.bias}_{args.assignment}.mat",
    #     "Deepmoji": f"data/projection_matrix/{args.experiment}/{args.model}/{args.bias}_{args.assignment}.mat"
    # }

# # Run code.
# for debiasing_method in ${Debiasing_method[@]}; do
#     for experiment in ${Experiments[@]}; do
#         for bias in ${Biases[@]}; do
#             for model in ${Models[@]}; do
#                 for assignment in ${Assignment_option[@]}; do
#                     experiment_id="tpr-gap_d-${debiasing_method}_m-${model}_e-${experiment}_a-${assignment}"
#                     echo ${experiment_id}
#                     python src/assignment/tpr-gap_political_exp.py \
#                         --persistent_dir ${persistent_dir} \
#                         --debiasing ${debiasing_method} \
#                         --model ${model} \
#                         --experiment ${experiment} \
#                         --assignment ${assignment} \
#                         --bias ${bias}
#                 done
#             done
#         done
#     done
# done
