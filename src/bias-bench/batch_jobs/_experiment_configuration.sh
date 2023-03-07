#!/bin/bash
# Variables for running batch jobs.

# Directory where all data and results are written.
persistent_dir="/bask/projects/j/jlxi8926-auto-sum/shun/code/unsup_svd/src/bias-bench/"

crows_models=(
    "bert"
    "albert"
    "roberta"
    "gpt2"
)

sal_lm_models=(
    "SALBertForMaskedLM"
    "SALAlbertForMaskedLM"
    "SALRobertaForMaskedLM"
    "SALGPT2LMHeadModel"
)

sal_models=(
    "SALBertModel"
    "SALAlbertModel"
    "SALRobertaModel" 
    "SALGPT2Model"
)

inlp_models=(
    "INLPBertModel"
    "INLPAlbertModel"
    "INLPRobertaModel"
    "INLPGPT2Model"
)

# Baseline models.
models=(
    "BertModel"
    "AlbertModel"
    "RobertaModel"
    "GPT2Model"
)

pre_assignment_types=(
    "Sal"
    "Kmeans"
    "Oracle"
    "partialSup"
)

bias_types=(
    "gender"
    "race"
    "religion"
)

# GLUE variables.
glue_tasks=(
    "cola"
    "mnli"
    "mrpc"
    "qnli"
    "qqp"
    "rte"
    "sst2"
    "stsb"
    "wnli"
)




# crows_models=(
#     "bert"
# )

# sal_lm_models=(
#     "SALBertForMaskedLM"
# )

# sal_models=(
#     "SALBertModel"
# )

# inlp_models=(
#     "INLPBertModel"
# )

# models=(
#     "BertModel"
# )

# pre_assignment_types=(
#     "Oracle"
# )

# bias_types=(
#     "religion"
# )

# # GLUE variables.
# glue_tasks=(
#     "cola"
# )


# Baseline masked language models.
masked_lm_models=(
    "BertForMaskedLM"
    "AlbertForMaskedLM"
    "RobertaForMaskedLM"
)

# Baseline causal language models.
causal_lm_models=(
    "GPT2LMHeadModel"
)

# Debiased masked language models.
sentence_debias_masked_lm_models=(
    "SentenceDebiasBertForMaskedLM"
    "SentenceDebiasAlbertForMaskedLM"
    "SentenceDebiasRobertaForMaskedLM"
)

inlp_masked_lm_models=(
    "INLPBertForMaskedLM"
    "INLPAlbertForMaskedLM"
    "INLPRobertaForMaskedLM"
)

cda_masked_lm_models=(
    "CDABertForMaskedLM"
    "CDAAlbertForMaskedLM"
    "CDARobertaForMaskedLM"
)

dropout_masked_lm_models=(
    "DropoutBertForMaskedLM"
    "DropoutAlbertForMaskedLM"
    "DropoutRobertaForMaskedLM"
)

self_debias_masked_lm_models=(
    "SelfDebiasBertForMaskedLM"
    "SelfDebiasAlbertForMaskedLM"
    "SelfDebiasRobertaForMaskedLM"
)

# Debiased causal language models.
sentence_debias_causal_lm_models=(
    "SentenceDebiasGPT2LMHeadModel"
)

inlp_causal_lm_models=(
    "INLPGPT2LMHeadModel"
)

cda_causal_lm_models=(
    "CDAGPT2LMHeadModel"
)

dropout_causal_lm_models=(
    "DropoutGPT2LMHeadModel"
)

self_debias_causal_lm_models=(
    "SelfDebiasGPT2LMHeadModel"
)

# Debiased base models.
sentence_debias_models=(
    "SentenceDebiasBertModel"
    "SentenceDebiasAlbertModel"
    "SentenceDebiasRobertaModel"
    "SentenceDebiasGPT2Model"
)



cda_models=(
    "CDABertModel"
    "CDAAlbertModel"
    "CDARobertaModel"
    "CDAGPT2Model"
)

dropout_models=(
    "DropoutBertModel"
    "DropoutAlbertModel"
    "DropoutRobertaModel"
    "DropoutGPT2Model"
)


declare -A sal_lm_models_to_sal_models=(
    ["SALBertForMaskedLM"]="SALBertModel"
    ["SALAlbertForMaskedLM"]="SALAlbertModel"
    ["SALRobertaForMaskedLM"]="SALRobertaModel"
    ["SALGPT2LMHeadModel"]="SALGPT2Model"
)

declare -A model_to_model_name_or_path=(
    ["BertModel"]="bert-base-uncased"
    ["AlbertModel"]="albert-base-v2"
    ["RobertaModel"]="roberta-base"
    ["GPT2Model"]="gpt2"
    ["BertForMaskedLM"]="bert-base-uncased"
    ["AlbertForMaskedLM"]="albert-base-v2"
    ["RobertaForMaskedLM"]="roberta-base"
    ["GPT2LMHeadModel"]="gpt2"
    ["BertForSequenceClassification"]="bert-base-uncased"
    ["AlbertForSequenceClassification"]="albert-base-v2"
    ["RobertaForSequenceClassification"]="roberta-base"
    ["GPT2ForSequenceClassification"]="gpt2"
    ["SentenceDebiasBertModel"]="bert-base-uncased"
    ["SentenceDebiasAlbertModel"]="albert-base-v2"
    ["SentenceDebiasRobertaModel"]="roberta-base"
    ["SentenceDebiasGPT2Model"]="gpt2"
    ["SentenceDebiasBertForMaskedLM"]="bert-base-uncased"
    ["SentenceDebiasAlbertForMaskedLM"]="albert-base-v2"
    ["SentenceDebiasRobertaForMaskedLM"]="roberta-base"
    ["SentenceDebiasGPT2LMHeadModel"]="gpt2"
    ["SentenceDebiasBertForSequenceClassification"]="bert-base-uncased"
    ["SentenceDebiasAlbertForSequenceClassification"]="albert-base-v2"
    ["SentenceDebiasRobertaForSequenceClassification"]="roberta-base"
    ["SentenceDebiasGPT2ForSequenceClassification"]="gpt2"
    ["INLPBertModel"]="bert-base-uncased"
    ["INLPAlbertModel"]="albert-base-v2"
    ["INLPRobertaModel"]="roberta-base"
    ["INLPGPT2Model"]="gpt2"
    ["INLPBertForMaskedLM"]="bert-base-uncased"
    ["INLPAlbertForMaskedLM"]="albert-base-v2"
    ["INLPRobertaForMaskedLM"]="roberta-base"
    ["INLPGPT2LMHeadModel"]="gpt2"
    ["INLPBertForSequenceClassification"]="bert-base-uncased"
    ["INLPAlbertForSequenceClassification"]="albert-base-v2"
    ["INLPRobertaForSequenceClassification"]="roberta-base"
    ["INLPGPT2ForSequenceClassification"]="gpt2"
    ["CDABertModel"]="bert-base-uncased"
    ["CDAAlbertModel"]="albert-base-v2"
    ["CDARobertaModel"]="roberta-base"
    ["CDAGPT2Model"]="gpt2"
    ["CDABertForMaskedLM"]="bert-base-uncased"
    ["CDAAlbertForMaskedLM"]="albert-base-v2"
    ["CDARobertaForMaskedLM"]="roberta-base"
    ["CDAGPT2LMHeadModel"]="gpt2"
    ["CDABertForSequenceClassification"]="bert-base-uncased"
    ["CDAAlbertForSequenceClassification"]="albert-base-v2"
    ["CDARobertaForSequenceClassification"]="roberta-base"
    ["CDAGPT2ForSequenceClassification"]="gpt2"
    ["DropoutBertModel"]="bert-base-uncased"
    ["DropoutAlbertModel"]="albert-base-v2"
    ["DropoutRobertaModel"]="roberta-base"
    ["DropoutGPT2Model"]="gpt2"
    ["DropoutBertForMaskedLM"]="bert-base-uncased"
    ["DropoutAlbertForMaskedLM"]="albert-base-v2"
    ["DropoutRobertaForMaskedLM"]="roberta-base"
    ["DropoutGPT2LMHeadModel"]="gpt2"
    ["DropoutBertForSequenceClassification"]="bert-base-uncased"
    ["DropoutAlbertForSequenceClassification"]="albert-base-v2"
    ["DropoutRobertaForSequenceClassification"]="roberta-base"
    ["DropoutGPT2ForSequenceClassification"]="gpt2"
    ["SelfDebiasBertForMaskedLM"]="bert-base-uncased"
    ["SelfDebiasAlbertForMaskedLM"]="albert-base-v2"
    ["SelfDebiasRobertaForMaskedLM"]="roberta-base"
    ["SelfDebiasGPT2LMHeadModel"]="gpt2"
    ["SALBertModel"]="bert-base-uncased"
    ["SALAlbertModel"]="albert-base-v2"
    ["SALRobertaModel"]="roberta-base"
    ["SALGPT2Model"]="gpt2"
    ["SALBertForMaskedLM"]="bert-base-uncased"
    ["SALAlbertForMaskedLM"]="albert-base-v2"
    ["SALRobertaForMaskedLM"]="roberta-base"
    ["SALGPT2LMHeadModel"]="gpt2"
    ["SALBertForSequenceClassification"]="bert-base-uncased"
    ["SALAlbertForSequenceClassification"]="albert-base-v2"
    ["SALRobertaForSequenceClassification"]="roberta-base"
    ["SALGPT2ForSequenceClassification"]="gpt2"
)


# For SentenceDebias and INLP, it is useful to have the base model
# that was used to compute the subspace or projection matrix.
declare -A debiased_model_to_base_model=(
    ["SentenceDebiasBertModel"]="BertModel"
    ["SentenceDebiasAlbertModel"]="AlbertModel"
    ["SentenceDebiasRobertaModel"]="RobertaModel"
    ["SentenceDebiasGPT2Model"]="GPT2Model"
    ["SentenceDebiasBertForMaskedLM"]="BertModel"
    ["SentenceDebiasAlbertForMaskedLM"]="AlbertModel"
    ["SentenceDebiasRobertaForMaskedLM"]="RobertaModel"
    ["SentenceDebiasGPT2LMHeadModel"]="GPT2Model"
    ["SentenceDebiasBertForSequenceClassification"]="BertModel"
    ["SentenceDebiasAlbertForSequenceClassification"]="AlbertModel"
    ["SentenceDebiasRobertaForSequenceClassification"]="RobertaModel"
    ["SentenceDebiasGPT2ForSequenceClassification"]="GPT2Model"
    ["INLPBertModel"]="BertModel"
    ["INLPAlbertModel"]="AlbertModel"
    ["INLPRobertaModel"]="RobertaModel"
    ["INLPGPT2Model"]="GPT2Model"
    ["INLPBertForMaskedLM"]="BertModel"
    ["INLPAlbertForMaskedLM"]="AlbertModel"
    ["INLPRobertaForMaskedLM"]="RobertaModel"
    ["INLPGPT2LMHeadModel"]="GPT2Model"
    ["INLPBertForSequenceClassification"]="BertModel"
    ["INLPAlbertForSequenceClassification"]="AlbertModel"
    ["INLPRobertaForSequenceClassification"]="RobertaModel"
    ["INLPGPT2ForSequenceClassification"]="GPT2Model"
    ["SALBertForMaskedLM"]="BertModel"
    ["SALAlbertForMaskedLM"]="AlbertModel"
    ["SALRobertaForMaskedLM"]="RobertaModel"
    ["SALGPT2LMHeadModel"]="GPT2Model"
    ["SALBertModel"]="BertModel"
    ["SALAlbertModel"]="AlbertModel"
    ["SALRobertaModel"]="RobertaModel"
    ["SALGPT2Model"]="GPT2Model"
)


declare -A debiased_model_to_masked_lm_model=(
    ["CDABertModel"]="BertForMaskedLM"
    ["CDAAlbertModel"]="AlbertForMaskedLM"
    ["CDARobertaModel"]="RobertaForMaskedLM"
    ["CDABertForMaskedLM"]="BertForMaskedLM"
    ["CDAAlbertForMaskedLM"]="AlbertForMaskedLM"
    ["CDARobertaForMaskedLM"]="RobertaForMaskedLM"
    ["CDAGPT2Model"]="GPT2LMHeadModel"
    ["CDAGPT2LMHeadModel"]="GPT2LMHeadModel"
    ["DropoutBertModel"]="BertForMaskedLM"
    ["DropoutAlbertModel"]="AlbertForMaskedLM"
    ["DropoutRobertaModel"]="RobertaForMaskedLM"
    ["DropoutGPT2Model"]="GPT2LMHeadModel"
    ["DropoutBertForMaskedLM"]="BertForMaskedLM"
    ["DropoutAlbertForMaskedLM"]="AlbertForMaskedLM"
    ["DropoutRobertaForMaskedLM"]="RobertaForMaskedLM"
    ["DropoutGPT2LMHeadModel"]="GPT2LMHeadModel"
    ["SALBertForSequenceClassification"]="SALBertForMaskedLM"
    ["SALAlbertForSequenceClassification"]="SALAlbertForMaskedLM"
    ["SALRobertaForSequenceClassification"]="SALRobertaForMaskedLM"
    ["SALGPT2ForSequenceClassification"]="SALGPT2LMHeadModel"
)

declare -A base_model_to_masked_lm_model=(
    ["SALBertModel"]="SALBertForMaskedLM"
    ["SALAlbertModel"]="SALAlbertForMaskedLM"
    ["SALRobertaModel"]="SALRobertaForMaskedLM"
    ["SALGPT2Model"]="SALGPT2LMHeadModel"
)

# StereoSet specific variables.
stereoset_score_types=(
    "likelihood"
)

stereoset_splits=(
    # "dev"
    "test"
)

# Types of representations to use for computing SentenceDebias subspace
# and INLP projection matrix.
representation_types=(
    "cls"
    "mean"
)


# SEAT variables.
# Space separated list of SEAT tests to run.
seat_tests="sent-religion1 "\
"sent-religion1b "\
"sent-religion2 "\
"sent-religion2b "\
"sent-angry_black_woman_stereotype "\
"sent-angry_black_woman_stereotype_b "\
"sent-weat3 "\
"sent-weat3b "\
"sent-weat4 "\
"sent-weat5 "\
"sent-weat5b "\
"sent-weat6 "\
"sent-weat6b "\
"sent-weat7 "\
"sent-weat7b "\
"sent-weat8 "\
"sent-weat8b"

declare -A model_to_n_classifiers=(["BertModel"]="80" ["AlbertModel"]="80" ["RobertaModel"]="80" ["GPT2Model"]="10")

