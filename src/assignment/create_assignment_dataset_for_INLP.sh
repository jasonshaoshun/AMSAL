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

python src/assignment/create_assignment_dataset_for_INLP.py