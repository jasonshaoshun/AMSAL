# model
# "INLPBertForMaskedLM"
# "INLPAlbertForMaskedLM"
# "INLPRobertaForMaskedLM"
# "INLPGPT2LMHeadModel"

# model -> model_name_or_path
# ["INLPBertForMaskedLM"]="bert-base-uncased"
# ["INLPAlbertForMaskedLM"]="albert-base-v2"
# ["INLPRobertaForMaskedLM"]="roberta-base"
# ["INLPGPT2LMHeadModel"]="gpt2"

# BertModel
# AlbertModel
# GPT2Model
# RobertaModel

import os
os.environ['TRANSFORMERS_CACHE'] = '/work/sc066/sc066/shunshao/code/bias-bench/experiments/cache_folder/'
import torch
import transformers

kwargs = {}
projection_matrix = torch.load()
kwargs["projection_matrix"] = '../../new_sal/data_in_mat/projection_matrix/AlbertModel/gender.pt'

# Load model and tokenizer. `load_path` can be used to override `model_name_or_path`.
model = getattr(models, "SALAlbertForMaskedLM")(
    args.model_name_or_path, **kwargs
)