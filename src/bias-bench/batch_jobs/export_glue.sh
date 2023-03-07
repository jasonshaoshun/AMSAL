source "batch_jobs/_experiment_configuration.sh"

for model in ${crows_models[@]}; do
    experiment_id="export-glue_m-${model}"
    echo ${experiment_id}
    python export/glue.py \
        --model_type ${model} \
        --checkpoint_dir "${persistent_dir}checkpoints" \
        --persistent_dir ${persistent_dir}
done