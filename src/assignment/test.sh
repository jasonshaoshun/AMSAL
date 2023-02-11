module load matlab

all_models=(
    "word_embeddings"
    "05_all"
    "FastText"
    "BERT"
)

for model_name in ${all_models[@]}; do
    matlab -r  "dataset_path = '../data_in_mat/${model_name}.mat'; model_name = '${model_name}'; bias_name = 'old_paper'; iter = 150; k = 2; batch_size_max = 30000; epoch_num = 3; batch_normal; exit;"
done

all_models=("BertModel" "AlbertModel" "RobertaModel" "GPT2Model")
all_bias_types=("gender" "race" "religion")

for model_name in ${all_models[@]}; do
    for bias_type in ${all_bias_types[@]}; do
        matlab -r  "dataset_path = '../../bias-bench/data/saved_dataset/${model_name}/${bias_type}'; model_name = '${model_name}'; bias_name = '${bias_type}'; iter = 150; k = 2; batch_size_max = 30000; epoch_num = 3; batch_normal; exit"
    done
done