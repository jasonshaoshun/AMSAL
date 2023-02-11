persistent_dir="/bask/projects/j/jlxi8926-auto-sum/shun/code/USAL/"

# Load the required modules
eval "$(/bask/projects/j/jlxi8926-auto-sum/shun/anaconda3/bin/conda shell.bash hook)"
conda activate ksal
export PYTHONPATH="${PYTHONPATH}:/bask/projects/j/jlxi8926-auto-sum/shun/code/USAL/"








declare -A experiment_to_sheet_name=(
    ["twitter-different-partial-n-concat"]="twit-Part-concat"
    ["twitter-different-partial-n-concat-v4"]="twit-Part-concat-v4"
    ["twitter-different-partial-n-short"]="twit-Part-short"
    ["twitter-concat-sentiment"]="twit-concat"
    ["twitter-concat-sentiment-v4"]="twit-concat-v4"
    ["twitter-short-sentiment"]="twit-short"
    ["twitter-labelled-by-overlap-concat"]="twit-WL-concat"
    ["twitter-labelled-by-overlap-concat-v4"]="twit-WL-concat-v4"
    ["twitter-labelled-by-overlap-short"]="twit-WL-short"
)

declare -A model_to_sheet_name=(
    ["BertModel"]="BT"
    ["FastText"]="FT"
)



# model="BertModel"
model="FastText"

bias="age-gender"
tex_folder="twitter"
summary_xlsx_path=${persistent_dir}/src/assignment/tables/summary.xlsx






# Run code.
Experiments_list=(
    # "twitter-concat-sentiment"
    # "twitter-short-sentiment"
    "twitter-concat-sentiment-v4"
)
summary_xlsx_path=${persistent_dir}/src/assignment/tables/summary.xlsx

for experiment in ${Experiments_list[@]}; do
    experiment_id="tpr-gap-tables_e-${experiment}_m-${model}"
    echo ${experiment_id}
    python src/assignment/export_tpr-gap_twitter.py \
        --persistent_dir ${persistent_dir} \
        --model ${model} \
        --tpr_gap_results_dir ${persistent_dir}/src/assignment/results/${experiment}/ \
        --save_path ${persistent_dir}/src/assignment/tables/${tex_folder}/tpr-table_e-${experiment}_m-${model}.tex \
        --sheet_name "${experiment_to_sheet_name[${experiment}]}_${model_to_sheet_name[${model}]}" \
        --summary_xlsx_path ${summary_xlsx_path}
done







# Experiments_list=(
#     # "twitter-different-partial-n-concat"
#     # "twitter-different-partial-n-short"
#     "twitter-different-partial-n-concat-v4"
# )

# for experiment in ${Experiments_list[@]}; do
#     experiment_id="tpr-gap-tables_e-${experiment}_m-${model}"
#     echo ${experiment_id}
#     echo ${experiment}
#     echo ${experiment_to_sheet_name[${experiment}]}
#     python src/assignment/export_tpr-gap_twitter_different_partial_n.py \
#         --persistent_dir ${persistent_dir} \
#         --model ${model} \
#         --tpr_gap_results_dir ${persistent_dir}/src/assignment/results/${experiment}/ \
#         --save_path ${persistent_dir}/src/assignment/tables/${tex_folder}/tpr-table_e-${experiment}_m-${model}.tex \
#         --sheet_name "${experiment_to_sheet_name[${experiment}]}_${model_to_sheet_name[${model}]}" \
#         --summary_xlsx_path ${summary_xlsx_path}
# done







# Run code.
Experiments_list=(
    # "twitter-labelled-by-overlap-concat"
    # "twitter-labelled-by-overlap-short"
    "twitter-labelled-by-overlap-concat-v4"
)

for experiment in ${Experiments_list[@]}; do
    experiment_id="tpr-gap-tables_e-${experiment}_m-${model}"
    echo ${experiment_id}
    python src/assignment/export_tpr-gap_twitter_labelled_by_overlap.py \
        --persistent_dir ${persistent_dir} \
        --model ${model} \
        --tpr_gap_results_dir ${persistent_dir}/src/assignment/results/${experiment}/ \
        --save_path ${persistent_dir}/src/assignment/tables/${tex_folder}/tpr-table_e-${experiment}_m-${model}.tex \
        --sheet_name "${experiment_to_sheet_name[${experiment}]}_${model_to_sheet_name[${model}]}" \
        --summary_xlsx_path ${summary_xlsx_path}
done
