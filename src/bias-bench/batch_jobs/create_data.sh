#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

experiment_id="create_data"
sbatch \
    --gres=gpu:1 \
    -c 4 \
    -J ${experiment_id} \
    -o ${persistent_dir}/logs/%x.%j.out \
    -e ${persistent_dir}/logs/%x.%j.err \
    python_job.sh experiments/create_data.py