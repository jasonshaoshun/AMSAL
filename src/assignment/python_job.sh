#!/bin/bash
#SBATCH --qos epsrc
#SBATCH --gpus 1
#SBATCH --time=24:00:00
# SBATCH --cpus-per-task 4

module purge
module load baskerville
module load CUDA/11.1.1-GCC-10.2.0

# Load the required modules
eval "$(/bask/projects/j/jlxi8926-auto-sum/shun/anaconda3/bin/conda shell.bash hook)"

conda activate ksal

# python -m venv --system-site-packages bench_environment
# source bench_environment/bin/activate
# python -m pip install -e .

export PYTHONPATH="${PYTHONPATH}:/bask/projects/j/jlxi8926-auto-sum/shun/code/USAL/"
persistent_dir="/bask/projects/j/jlxi8926-auto-sum/shun/code/USAL/"

# Run code.
python -u "$@"