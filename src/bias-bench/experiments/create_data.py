import argparse
import os
import sys
import torch
import transformers
import numpy as np
import sklearn
import scipy.io as io
from sklearn.svm import LinearSVC
from tqdm import tqdm
from bias_bench.dataset import load_inlp_data
from bias_bench.debias.inlp import compute_projection_matrix
from bias_bench.debias.inlp import debias
from bias_bench.model import models
from bias_bench.debias.inlp import debias
import nltk

nltk.download('punkt')
parser = argparse.ArgumentParser(description="Create bias-bench data.")

parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    help="Directory where all persistent data will be stored.",
)

parser.add_argument(
    "--folder_name",
    action="store",
    type=str,
    help="The name of the folder used to store bias-bench datasets",
)


def _extract_gender_features(
    model,
    tokenizer,
    male_sentences,
    female_sentences,
    neutral_sentences,
    device,
):
    """Encodes gender sentences to create a set of representations to train classifiers
    for INLP on.
    Notes:
        * Implementation taken from  https://github.com/pliang279/LM_bias.
    """
    model.to(device)

    male_features = []
    female_features = []
    neutral_features = []

    # Encode the sentences.
    with torch.no_grad():
        for sentence in tqdm(male_sentences, desc="Encoding male sentences"):
            input_ids = tokenizer(
                sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
            ).to(device)

            outputs = model(**input_ids)["last_hidden_state"]
            outputs = torch.mean(outputs, dim=1)
            outputs = outputs.squeeze().detach().cpu().numpy()

            male_features.append(outputs)

        for sentence in tqdm(female_sentences, desc="Encoding female sentences"):
            input_ids = tokenizer(
                sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
            ).to(device)

            outputs = model(**input_ids)["last_hidden_state"]
            outputs = torch.mean(outputs, dim=1)
            outputs = outputs.squeeze().detach().cpu().numpy()

            female_features.append(outputs)

        for sentence in tqdm(neutral_sentences, desc="Encoding neutral sentences"):
            input_ids = tokenizer(
                sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
            ).to(device)

            outputs = model(**input_ids)["last_hidden_state"]
            outputs = torch.mean(outputs, dim=1)
            outputs = outputs.squeeze().detach().cpu().numpy()

            neutral_features.append(outputs)

    male_features = np.array(male_features)
    female_features = np.array(female_features)
    neutral_features = np.array(neutral_features)

    return male_features, female_features, neutral_features


def _extract_binary_features(model, tokenizer, bias_sentences, neutral_sentences):
    """Encodes race/religion sentences to create a set of representations to train classifiers
    for INLP on.
    Notes:
        * Sentences are split into two classes based upon if they contain *any* race/religion bias
          attribute words.
    """
    model.to(device)

    bias_features = []
    neutral_features = []

    # Encode the sentences.
    with torch.no_grad():
        for sentence in tqdm(bias_sentences, desc="Encoding bias sentences"):
            input_ids = tokenizer(
                sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
            ).to(device)

            outputs = model(**input_ids)["last_hidden_state"]
            outputs = torch.mean(outputs, dim=1)
            outputs = outputs.squeeze().detach().cpu().numpy()

            bias_features.append(outputs)

        for sentence in tqdm(neutral_sentences, desc="Encoding neutral sentences"):
            input_ids = tokenizer(
                sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
            ).to(device)

            outputs = model(**input_ids)["last_hidden_state"]
            outputs = torch.mean(outputs, dim=1)
            outputs = outputs.squeeze().detach().cpu().numpy()

            neutral_features.append(outputs)

    bias_features = np.array(bias_features)
    neutral_features = np.array(neutral_features)

    return bias_features, neutral_features


def _split_gender_dataset(male_feat, female_feat, neut_feat):
    np.random.seed(0)

    X = np.concatenate((male_feat, female_feat, neut_feat), axis=0)

    y_male = np.ones(male_feat.shape[0], dtype=int)
    y_female = np.zeros(female_feat.shape[0], dtype=int)
    y_neutral = -np.ones(neut_feat.shape[0], dtype=int)

    y = np.concatenate((y_male, y_female, y_neutral))

    X_train_dev, X_test, y_train_dev, Y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(
        X_train_dev, y_train_dev, test_size=0.3, random_state=0
    )

    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test


def _split_binary_dataset(bias_feat, neut_feat):
    np.random.seed(0)

    X = np.concatenate((bias_feat, neut_feat), axis=0)

    y_bias = np.ones(bias_feat.shape[0], dtype=int)
    y_neutral = np.zeros(neut_feat.shape[0], dtype=int)

    y = np.concatenate((y_bias, y_neutral))

    X_train_dev, X_test, y_train_dev, Y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(
        X_train_dev, y_train_dev, test_size=0.3, random_state=0
    )

    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test


def create_dataset(model, tokenizer, data, bias_type, device, n_classifiers=80):
    """Runs INLP.
    Notes:
        * We use the same classifier hyperparameters as Liang et al.
    Args:
        model: HuggingFace model (e.g., BertModel) to compute the projection
            matrix for.
        tokenizer: HuggingFace tokenizer (e.g., BertTokenizer). Used to pre-process
            examples for the INLP classifiers.
        data (`dict`): Dictionary of sentences used to train the INLP classifiers.
        bias_type (`str`): Type of bias to compute a projection matrix for.
        n_classifiers (`int`): How many classifiers to train when computing INLP
            projection matrix.
    """
    if bias_type == "gender":
        male_sentences = data["male"]
        female_sentences = data["female"]
        neutral_sentences = data["neutral"]

        male_features, female_features, neutral_features = _extract_gender_features(
            model, tokenizer, male_sentences, female_sentences, neutral_sentences, device
        )

        X_train, X_dev, X_test, Y_train, Y_dev, Y_test = _split_gender_dataset(
            male_features, female_features, neutral_features
        )

    else:
        bias_sentences = data["bias"]
        neutral_sentences = data["neutral"]

        bias_features, neutral_features = _extract_binary_features(
            model, tokenizer, bias_sentences, neutral_sentences
        )

        X_train, X_dev, X_test, Y_train, Y_dev, Y_test = _split_binary_dataset(
            bias_features, neutral_features
        )

    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test

if __name__ == "__main__":
    args = parser.parse_args()

    nltk.download('punkt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # all_models=["BertModel", "AlbertModel", "RobertaModel", "GPT2Model"]
    all_models=["BertModel"]
    # all_bias_types=["gender", "race", "religion"]
    all_bias_types=["religion"]

    seed = 0
    for model_name in all_models:
        for bias_type in all_bias_types:
            if model_name == "GPT2Model":
                n_classifiers = "10"
            else:
                n_classifiers = "80"
            
            if model_name == "BertModel":
                model_name_path = "bert-base-uncased"
            elif model_name == "AlbertModel":
                model_name_path = "albert-base-v2"
            elif model_name == "RobertaModel":
                model_name_path = "roberta-base"
            elif model_name == "GPT2Model":
                model_name_path = "gpt2"
        
            args = parser.parse_args(sys.argv)

            print("Computing projection matrix:")
            print(f" - model: {model_name}")
            print(f" - model_name_path: {model_name_path}")
            print(f" - bias_type: {bias_type}")
            print(f" - n_classifiers: {n_classifiers}")
            print(f" - seed: {seed}")

            # Load data for INLP classifiers.
            data = load_inlp_data(args.persistent_dir, bias_type, seed=seed)

            # Load model and tokenizer.
            model = getattr(models, model_name)(model_name_path)
            model.eval()
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_path)
            x_train, x_dev, x_test, y_train, y_dev, y_test = create_dataset(
                model,
                tokenizer,
                data,
                bias_type,
                device,
                n_classifiers
            )

            np.savez(f"data/{args.folder_name}/{model_name}/{bias_type}.npz", \
                    x_train = x_train, x_dev = x_dev, x_test = x_test, \
                    y_p_train = y_train, y_p_dev = y_dev, y_p_test = y_test)
            
            io.savemat(f"data/{args.folder_name}/{model_name}/{bias_type}.mat", \
                    mdict={'x_train': x_train, 'x_dev': x_dev, 'x_test': x_test, \
                            'y_p_train': y_train, 'y_p_dev': y_dev, 'y_p_test': y_test})

            n_train = x_train.shape[0]
            for i in range(n_train):
                if np.argwhere(np.all(x_train == x_train[i, :], axis=1) == 1).size != 1:
                    print(f"more than one occurences: i is {i}, index is {np.argwhere(np.all(x_train == x_train[i, :], axis=1) == 1)}")
                    break