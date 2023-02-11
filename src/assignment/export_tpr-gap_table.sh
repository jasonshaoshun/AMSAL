persistent_dir="/bask/projects/j/jlxi8926-auto-sum/shun/code/USAL/"

# Load the required modules
eval "$(/bask/projects/j/jlxi8926-auto-sum/shun/anaconda3/bin/conda shell.bash hook)"
conda activate ksal
export PYTHONPATH="${PYTHONPATH}:/bask/projects/j/jlxi8926-auto-sum/shun/code/USAL/"

declare -A experiment_to_sheet_name=(
    ["ratio_on_race_0.5"]="R5S5"
    ["ratio_on_race_0.8"]="R8S5"
    ["ratio_on_sent_0.8"]="R5S8"
)




model="deepmoji"
experiment="deepmoji"
tex_folder="deepmoji"
summary_xlsx_path=${persistent_dir}/src/assignment/tables/summary.xlsx

# Run code.
variable_with_changing_ratio="ratio_on_race"
AM_Ratios=(
    "0.8"
    "0.5"
)

for assignment_ratio in ${AM_Ratios[@]}; do
    experiment_id="tpr-gap-tables_m-${model}_r-${assignment_ratio}"
    echo ${experiment_id}
    python src/assignment/export_tpr-gap_table.py \
        --persistent_dir ${persistent_dir} \
        --model ${model} \
        --tpr_gap_results_dir ${persistent_dir}/src/assignment/results/${experiment}/${variable_with_changing_ratio}  \
        --save_path ${persistent_dir}/src/assignment/tables/${tex_folder}/tpr-table_v-${variable_with_changing_ratio}_m-${model}_r-${assignment_ratio}.tex \
        --assignment_ratio ${assignment_ratio} \
        --sheet_name ${experiment_to_sheet_name["${variable_with_changing_ratio}_${assignment_ratio}"]} \
        --summary_xlsx_path ${summary_xlsx_path}
done


# Run code.
variable_with_changing_ratio="ratio_on_sent"
AM_Ratios=(
    "0.8"
)

for assignment_ratio in ${AM_Ratios[@]}; do
    experiment_id="tpr-gap-tables_e-${experiment}_m-${model}_r-${assignment_ratio}"
    echo ${experiment_id}
    python src/assignment/export_tpr-gap_table.py \
        --persistent_dir ${persistent_dir} \
        --model ${model} \
        --tpr_gap_results_dir ${persistent_dir}/src/assignment/results/${experiment}/${variable_with_changing_ratio}  \
        --save_path ${persistent_dir}/src/assignment/tables/${tex_folder}/tpr-table_v-${variable_with_changing_ratio}_m-${model}_r-${assignment_ratio}.tex \
        --assignment_ratio ${assignment_ratio} \
        --sheet_name ${experiment_to_sheet_name["${variable_with_changing_ratio}_${assignment_ratio}"]} \
        --summary_xlsx_path ${summary_xlsx_path}
done



# Run code.
variable_with_changing_ratio="ratio_on_sent"
AM_Ratios=(
    "0.8"
)

for assignment_ratio in ${AM_Ratios[@]}; do
    experiment_id="tpr-gap-tables_e-${experiment}_m-${model}_r-${assignment_ratio}"
    echo ${experiment_id}
    python src/assignment/export_tpr-gap_table.py \
        --persistent_dir ${persistent_dir} \
        --model ${model} \
        --tpr_gap_results_dir ${persistent_dir}/src/assignment/results/${experiment}/${variable_with_changing_ratio}  \
        --save_path ${persistent_dir}/src/assignment/tables/${tex_folder}/tpr-table_v-${variable_with_changing_ratio}_m-${model}_r-${assignment_ratio}.tex \
        --assignment_ratio ${assignment_ratio} \
        --sheet_name ${experiment_to_sheet_name["${variable_with_changing_ratio}_${assignment_ratio}"]} \
        --summary_xlsx_path ${summary_xlsx_path}
done



