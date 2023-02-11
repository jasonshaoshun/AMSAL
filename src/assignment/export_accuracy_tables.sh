persistent_dir="/bask/projects/j/jlxi8926-auto-sum/shun/code/unsup_svd/"

Models=(
    "Bert"
    "Fasttext"
    "Deepmoji"
)

all_assignments=(
    "Bert"
    "Fasttext"
    "Deepmoji"
)

# Model_to_experiment=(
#     ["Bert"]="ProfessionGender"
#     ["Fasttext"]="ProfessionGender"
#     ["Deepmoji"]="SentimentRace"
# )


# Load the required modules
eval "$(/bask/projects/j/jlxi8926-auto-sum/shun/anaconda3/bin/conda shell.bash hook)"
conda activate ksal
export PYTHONPATH="${PYTHONPATH}:/bask/projects/j/jlxi8926-auto-sum/shun/code/unsup_svd/"


# Run code.
for model in ${Models[@]}; do
    experiment_id="tpr-gap-tables_m-${model}"
    echo ${experiment_id}
    python src/assignment/export_accuracy_tables.py \
        --persistent_dir ${persistent_dir} \
        --model ${model} \
        --assignment_range Kmeans partialSup Sal
done