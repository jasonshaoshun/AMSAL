persistent_dir="/bask/projects/j/jlxi8926-auto-sum/shun/code/unsup_svd/"


Models=(
    # "Bert"
    # "Fasttext"
    # "Deepmoji05"
    # "Deepmoji08"
    # "Deepmoji80"
    "BertModel"
)


Model_to_experiment=(
    ["Bert"]="ProfessionGender"
    ["Fasttext"]="ProfessionGender"
    ["Deepmoji05"]="SentimentRace"
    ["Deepmoji08"]="SentimentRace"
    ["Deepmoji80"]="SentimentRace"
    ["BertModel"]="SentimentAgeGender"
)

tpr_gap_results_dir="twitter-different-partial-n/tpr-gap/"

# Load the required modules
eval "$(/bask/projects/j/jlxi8926-auto-sum/shun/anaconda3/bin/conda shell.bash hook)"
conda activate ksal
export PYTHONPATH="${PYTHONPATH}:/bask/projects/j/jlxi8926-auto-sum/shun/code/unsup_svd/"


# Run code.
for model in ${Models[@]}; do
    experiment_id="tpr-gap-tables_m-${model}"
    echo ${experiment_id}
    python src/assignment/export_tpr-gap_table_political_exp.py \
        --persistent_dir ${persistent_dir} \
        --model ${model} \
        --tpr_gap_results_dir ${tpr_gap_results_dir} \
        --experiment ${Model_to_experiment[${model}]}
done