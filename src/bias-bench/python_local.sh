source batch_jobs/_experiment_configuration.sh

# python -m venv --system-site-packages bench_environment
# source bench_environment/bin/activate
# python -m pip install -e .

export PYTHONPATH="${PYTHONPATH}:/bask/projects/j/jlxi8926-auto-sum/shun/code/unsup_svd/src/bias-bench"
export TRANSFORMERS_CACHE=/bask/projects/j/jlxi8926-auto-sum/shun/code/unsup_svd/src/bias-bench/cache
export HF_DATASETS_CACHE=/bask/projects/j/jlxi8926-auto-sum/shun/code/unsup_svd/src/bias-bench/cache
# Run code.
python -u "$@" --persistent_dir ${persistent_dir}
