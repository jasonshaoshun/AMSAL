#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

sbatch \
    --gres=gpu:1 \
    -o ${persistent_dir}/logs/%j.out \
    -e ${persistent_dir}/logs/%j.err \
    python_job.sh experiments/stereoset_evaluation.py \
        --predictions_dir "${persistent_dir}/results/stereoset/" \
        --output_file "${persistent_dir}/results/stereoset/results.json"
