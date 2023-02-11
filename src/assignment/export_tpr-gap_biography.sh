persistent_dir="/bask/projects/j/jlxi8926-auto-sum/shun/code/USAL/"

# Load the required modules
eval "$(/bask/projects/j/jlxi8926-auto-sum/shun/anaconda3/bin/conda shell.bash hook)"
conda activate ksal
export PYTHONPATH="${PYTHONPATH}:/bask/projects/j/jlxi8926-auto-sum/shun/code/USAL/"

declare -A experiment_to_sheet_name=(
    ["biography_BertModel"]="bio-Bert"
    ["biography_FastText"]="bio-Fast"
    ["biography-labelled-by-overlap_BertModel"]="bio-overlap-Bert"
    ["biography-labelled-by-overlap_FastText"]="bio-overlap-Fast"
    ["biography-different-partial-n_BertModel"]="bio-dif-Partial-Bert"
    ["biography-different-partial-n_FastText"]="bio-dif-Partial-Fast"
)


tex_folder="biography"
summary_xlsx_path=${persistent_dir}/src/assignment/tables/summary.xlsx


Models=(
    "BertModel"
    "FastText"
)
bias="gender"



experiment="biography"
for model in ${Models[@]}; do
    experiment_id="tpr-gap-tables_e-${experiment}_m-${model}"
    echo ${experiment_id}
    echo ${experiment_to_sheet_name["${experiment}_${model}"]}
    python src/assignment/export_tpr-gap_biography.py \
        --persistent_dir ${persistent_dir} \
        --model ${model} \
        --tpr_gap_results_dir ${persistent_dir}/src/assignment/results/${experiment}/${model}  \
        --save_path ${persistent_dir}/src/assignment/tables/${tex_folder}/tpr-table_e-${experiment}_m-${model}.tex \
        --sheet_name ${experiment_to_sheet_name["${experiment}_${model}"]} \
        --summary_xlsx_path ${summary_xlsx_path}
done






# experiment="biography-different-partial-n"

# for model in ${Models[@]}; do
#     experiment_id="tpr-gap-tables_e-${experiment}_m-${model}"
#     echo ${experiment_id}
#     echo ${experiment_to_sheet_name["${experiment}_${model}"]}
#     python src/assignment/export_tpr-gap_biography_different_partial_n.py \
#         --persistent_dir ${persistent_dir} \
#         --model ${model} \
#         --tpr_gap_results_dir ${persistent_dir}/src/assignment/results/${experiment}/${model}  \
#         --save_path ${persistent_dir}/src/assignment/tables/${tex_folder}/tpr-table_e-${experiment}_m-${model}.tex \
#         --sheet_name ${experiment_to_sheet_name["${experiment}_${model}"]} \
#         --summary_xlsx_path ${summary_xlsx_path}
# done






experiment="biography-labelled-by-overlap"

for model in ${Models[@]}; do
    experiment_id="tpr-gap-tables_e-${experiment}_m-${model}"
    echo ${experiment_id}
    echo ${experiment_to_sheet_name["${experiment}_${model}"]}
    python src/assignment/export_tpr-gap_biography_labelled_by_overlap.py \
        --persistent_dir ${persistent_dir} \
        --model ${model} \
        --tpr_gap_results_dir ${persistent_dir}/src/assignment/results/${experiment}/${model}  \
        --save_path ${persistent_dir}/src/assignment/tables/${tex_folder}/tpr-table_e-${experiment}_m-${model}.tex \
        --sheet_name ${experiment_to_sheet_name["${experiment}_${model}"]} \
        --summary_xlsx_path ${summary_xlsx_path}
done