#!/bin/bash
#SBATCH --qos epsrc
#SBATCH --gpus 1 \
#SBATCH --time=2:00:00

module purge
module load baskerville
module load CUDA/11.1.1-GCC-10.2.0

# Load the required modules
eval "$(/bask/projects/j/jlxi8926-auto-sum/shun/anaconda3/bin/conda shell.bash hook)"
conda activate biasbench

source batch_jobs/_experiment_configuration.sh

echo "Host - $HOSTNAME"
echo "Commit - $(git rev-parse HEAD)"
nvidia-smi

nvcc -V

# python -m venv --system-site-packages bench_environment
# source bench_environment/bin/activate
# python -m pip install -e .

export PYTHONPATH="${PYTHONPATH}:/bask/projects/j/jlxi8926-auto-sum/shun/code/unsup_svd/src/bias-bench"
export TRANSFORMERS_CACHE=/bask/projects/j/jlxi8926-auto-sum/shun/code/unsup_svd/src/bias-bench/cache
export HF_DATASETS_CACHE=/bask/projects/j/jlxi8926-auto-sum/shun/code/unsup_svd/src/bias-bench/cache

# Run code.
python -u "$@" --persistent_dir ${persistent_dir}
