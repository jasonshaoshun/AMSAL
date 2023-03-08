source "batch_jobs/_experiment_configuration.sh"

for model in ${crows_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="export-stereoset_m-${model}_t-${bias_type}"
        echo ${experiment_id}
        python export/stereoset.py \
            --model_type ${model} \
            --bias_type ${bias_type} \
            --persistent_dir ${persistent_dir}
    done
done
