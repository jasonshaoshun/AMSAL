import pandas as pd
import torch
import transformers
from tqdm import tqdm
import numpy as np
import scipy.io as io
import pickle
from gensim.models import KeyedVectors
import re
import time
from string import punctuation
import random
import json
import argparse

parser = argparse.ArgumentParser(description="Run the tpr-gap experiments in INLP")

parser.add_argument(
    "--version",
    action="store",
    type=str,
    help="Version of data",
)


def one_hot_to_1d(one_hot_biases_train, one_hot_biases_dev):
    dim = one_hot_biases_train.shape[1]
    if dim == 2:
        biases_train = map(lambda num: 1 if (num==[1, 0]).all() else 2, one_hot_biases_train)
        biases_train = np.asarray(list(biases_train))

        biases_dev = map(lambda num: 1 if (num==[1, 0]).all() else 2, one_hot_biases_dev)
        biases_dev = np.asarray(list(biases_dev))

    elif dim == 3:
        biases_train = map(lambda num: 1 if (num==[1, 0, 0]).all() else (2 if (num==[0, 1, 0]).all() \
        else 3), one_hot_biases_train)
        biases_train = np.asarray(list(biases_train))

        biases_dev = map(lambda num: 1 if (num==[1, 0, 0]).all() else (2 if (num==[0, 1, 0]).all() \
        else 3), one_hot_biases_dev)
        biases_dev = np.asarray(list(biases_dev))

    elif dim == 5:
        biases_train = map(lambda num: 1 if (num==[1, 0, 0, 1, 0]).all() else (2 if (num==[1, 0, 0, 0, 1]).all() \
        else (3 if (num==[0, 1, 0, 1, 0]).all() else(4 if (num==[0, 1, 0, 0, 1]).all() \
        else (5 if (num==[0, 0, 1, 1, 0]).all() else 6)))), one_hot_biases_train)
        biases_train = np.asarray(list(biases_train))

        biases_dev = map(lambda num: 1 if (num==[1, 0, 0, 1, 0]).all() else (2 if (num==[1, 0, 0, 0, 1]).all() \
        else (3 if (num==[0, 1, 0, 1, 0]).all() else(4 if (num==[0, 1, 0, 0, 1]).all() \
        else (5 if (num==[0, 0, 1, 1, 0]).all() else 6)))), one_hot_biases_dev)
        biases_dev = np.asarray(list(biases_dev))

    return biases_train, biases_dev


def parse_json(json_path):
    male_words = []
    female_words = []
    f = open(json_path)
    rows = json.load(f)
    for row in rows:
        if row["gender"] == "m":
            male_words.append(row["word"])
        if row["gender"] == "f":
            female_words.append(row["word"])
        if "gender_map" in row:
            if "m" in row["gender_map"]:
                male_words.append(row["gender_map"]["m"][0]["word"])
            if "f" in row["gender_map"]:
                female_words.append(row["gender_map"]["f"][0]["word"])
    return set(male_words), set(female_words)



# Predicting if a sentence is written by a male or a female
def gender_the_sentence(sentence, male_words, female_words):
    sentence = sentence.lower()
    sentence_words = sentence.split(" ")
    sentence_words = [w.strip(punctuation) for w in sentence_words
                      if len(w.strip(punctuation)) > 0]
    mw_length = len(male_words.intersection(sentence_words))
    fw_length = len(female_words.intersection(sentence_words))
    if mw_length > fw_length:
        is_female = 0
    elif mw_length < fw_length:
        is_female = 1
    else:
        is_female = -1
    return is_female


# Iterating over all the sentences and printing a summary of the predictions
def gender_the_dataset(sentences, male_words, female_words):
    predicted_genders = []
    gender_count = [0, 0, 0]
    for sentence in sentences:
        predicted_gender = gender_the_sentence(sentence, male_words, female_words)
        gender_count[predicted_gender+1] += 1
        predicted_genders.append(predicted_gender)
    print("There are {} unknowns, {} males, and {} females in the original dataset".format(
        gender_count[0], gender_count[1],gender_count[2]))
    return predicted_genders


# Replace all the unknown values by a coin toss
def create_dataset(predicted_genders, random_seed):
    editable_predicted_genders = predicted_genders.copy()
    random.seed(random_seed)
    length = len(predicted_genders)
    
    for i in range(length):
        if predicted_genders[i] == -1:
            flip = random.randint(0, 1)
            editable_predicted_genders[i] = flip
    return editable_predicted_genders


# generating a list of datasets
def generate_datasets(sentences, male_words, female_words, datasets_num):
    datasets = []
    predicted_genders = gender_the_dataset(sentences, male_words, female_words)
    for i in range(datasets_num):
        dataset = create_dataset(predicted_genders, i)
        datasets.append(dataset)
        print("There are {} unknowns, {} males, and {} females in dataset_{}".format(0, sum(dataset),
                                                                                       len(dataset) - sum(dataset), i))
    return datasets


def load_word_vectors(fname):
    
    model = KeyedVectors.load_word2vec_format(fname, binary=False)
    # vecs = model.vectors
    # # words = list(model.vocab.keys())
    # words = list(model.index_to_key)
    # return model, vecs, words
    return model




if __name__ == "__main__":
    args = parser.parse_args()
    start = time.time()

    # k = 10

    model_name = "FastText"
    # model_name = "BertModel"
    # device = "cpu"
    # # device = "cuda"
    # model_name_or_path = "bert-base-uncased"
    # model = transformers.BertModel.from_pretrained(model_name_or_path)
    # model.eval()
    # model.to(device)
    # tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    file = open(f"data/twitter/df_con_sen_{args.version}", 'rb')
    data = pickle.load(file)
    # close the file
    file.close()

    # data = data[:100]
    train_n = int(len(data) * 0.7)

    # Load gendered words
    datasets_num = 10
    json_path = 'data/twitter/gendered_words.json'
    male_words, female_words = parse_json(json_path)

    # Check overlap between context and gender attribute words, then create new labels
    gender_label_by_overlap = generate_datasets(data["db_text"].copy(), male_words, female_words, datasets_num)


    word2vec_model = load_word_vectors("data/FastText/crawl-300d-2M.vec")

    tweet_sentences = data["db_text"].copy()
    tweet_encoding = []
    for sentence in tqdm(tweet_sentences, desc="Encoding tweet_sentences"):
        bagofwords = np.sum([word2vec_model[w] if w in word2vec_model else word2vec_model["unk"] for w in sentence], axis = 0)
        tweet_encoding.append(bagofwords)
            
    twitter_concat = np.asarray(tweet_encoding)
    twitter_concat_train = twitter_concat[:train_n, :]
    twitter_concat_dev =twitter_concat[train_n:, :]



    tweet_sentences = data["Tweet"].copy()
    tweet_encoding = []
    for sentence in tqdm(tweet_sentences, desc="Encoding tweet_sentences"):
        bagofwords = np.sum([word2vec_model[w] if w in word2vec_model else word2vec_model["unk"] for w in sentence], axis = 0)
        tweet_encoding.append(bagofwords)

    twitter_short = np.asarray(tweet_encoding)
    twitter_short_train = twitter_short[:train_n, :]
    twitter_short_dev = twitter_short[train_n:, :]


    # location = data["locations"].copy()

    # location_bert = []
    # with torch.no_grad():
    #     for sentence in tqdm(location, desc="Encoding locations"):
            
    #         input_ids = tokenizer(
    #             sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
    #         ).to(device)

    #         outputs = model(**input_ids)["last_hidden_state"]
    #         outputs = torch.mean(outputs, dim=1)
    #         outputs = outputs.squeeze().detach().cpu().numpy()

    #         location_bert.append(outputs)
            
    # location = np.asarray(location_bert)

    # # Reduce the dimension of biases by SVD
    # u, s, vh = np.linalg.svd(location, full_matrices=False)
    # s_updated = np.zeros((k, k))
    # np.fill_diagonal(s_updated, s[:k])
    # location_concat = np.dot(u[:, :k], s_updated)



    age_group = np.asarray(data["age_group"])

    ######################################################################################
    # Encode Y1: Age 
    one_hot_age = map(lambda num: [1, 0, 0] if (num==0) else ([0, 1, 0] if (num==1) else [0, 0, 1]), age_group)
    one_hot_age = np.asarray(list(one_hot_age))
    one_hot_age_train = one_hot_age[:train_n, :]
    one_hot_age_dev = one_hot_age[train_n:, :]





    ######################################################################################
    # Encode Y2: Gender True

    gender_group = np.asarray(data["gender_group"])

    one_hot_gender = map(lambda num: [0, 1] if (num==0) else [1, 0], gender_group)
    one_hot_gender = np.asarray(list(one_hot_gender))

    one_hot_gender_train_true = one_hot_gender[:train_n, :]
    one_hot_gender_dev_true = one_hot_gender[train_n:, :]





    ######################################################################################
    # Encode Y2: Gender by overlap checking

    gender_group = np.asarray(gender_label_by_overlap)

    one_hot_gender = []
    for i in range(10):
        one_hot_gender.append(list(map(lambda num: [0, 1] if (num==0) else [1, 0], gender_group[i])))
        
    one_hot_gender = np.asarray(one_hot_gender)
    one_hot_gender_train = one_hot_gender[:, :train_n, :]
    one_hot_gender_dev = one_hot_gender[:, train_n:, :]
    print(f"one_hot_gender_train shape {one_hot_gender_train.shape}, one_hot_gender_dev shape {one_hot_gender_dev.shape}")


    # ######################################################################################
    # # Encode YM: Sentiment
    # sentiment = np.asarray(data["sentiment"])
    # sentiment = np.expand_dims(sentiment, axis=1)
    # sentiment_train = sentiment[:train_n, :]
    # sentiment_dev = sentiment[train_n:, :]

    # sentiment_one_hot = map(lambda num: [1, 0, 0] if (num==0) else ([0, 1, 0] if (num==1) \
    #                         else ([0, 0, 1])), sentiment)
    # sentiment_one_hot = np.asarray(list(sentiment_one_hot))
    # sentiment_one_hot_train = sentiment_one_hot[:train_n, :]
    # sentiment_one_hot_dev = sentiment_one_hot[train_n:, :]


    ######################################################################################
    # Encode YM: vulgar
    majority = np.asarray(data["Majority"])
    majority = np.expand_dims(majority, axis=1)
    majority_train = majority[:train_n, :]
    majority_dev = majority[train_n:, :]

    majority_one_hot = map(lambda num: [1, 0, 0, 0, 0] if (num==1) else ([0, 1, 0, 0, 0] if (num==2) \
                            else ([0, 0, 1, 0, 0] if (num==3) else([0, 0, 0, 1, 0] if (num==4) \
                            else [0, 0, 0, 0, 1]))), majority)
    majority_one_hot = np.asarray(list(majority_one_hot))
    majority_one_hot_train = majority_one_hot[:train_n, :]
    majority_one_hot_dev = majority_one_hot[train_n:, :]
















    # Few things to change before starting whole experiments:
    # 1.keep eyes on x_train, it should be =Tweet on normal twitter and =db_text on concat twitter.









    # y_m_train = sentiment_train
    # y_m_dev = sentiment_dev
    # y_m_train_one_hot = sentiment_one_hot_train
    # y_m_dev_one_hot = sentiment_one_hot_dev

    # y_majority_train = majority_train
    # y_majority_dev = majority_dev
    # y_majority_train_one_hot = majority_one_hot_train
    # y_majority_dev_one_hot = majority_one_hot_dev

    y_m_train = majority_train
    y_m_dev = majority_dev
    y_m_train_one_hot = majority_one_hot_train
    y_m_dev_one_hot = majority_one_hot_dev


    one_hot_biases_train = np.concatenate((one_hot_age_train, one_hot_gender_train_true), axis=1)
    one_hot_biases_dev = np.concatenate((one_hot_age_dev, one_hot_gender_dev_true), axis=1)
    biases_train, biases_dev = one_hot_to_1d(one_hot_biases_train, one_hot_biases_dev)

    # Save the dataset by age and gender
    bias_name = "age-gender"


    experiment_name = f"twitter-labelled-by-overlap-concat-{args.version}"
    x_AM_train = twitter_concat_train
    x_AM_dev = twitter_concat_dev
    x_tpr_train = twitter_short_train
    x_tpr_dev = twitter_short_dev

    np.savez(f"data/{experiment_name}/{model_name}/{bias_name}.npz", x_train = x_tpr_train, x_dev = x_tpr_dev, \
        y_p_train = biases_train, y_p_dev = biases_dev, \
        y_m_train = y_m_train, y_m_dev = y_m_dev, \
        y_m_train_one_hot = y_m_train_one_hot, y_m_dev_one_hot = y_m_dev_one_hot)

    for i in range(10):
        one_hot_biases_train = np.concatenate((one_hot_age_train, one_hot_gender_train[i, :]), axis=1)
        one_hot_biases_dev = np.concatenate((one_hot_age_dev, one_hot_gender_dev[i, :]), axis=1)

        io.savemat(f"data/{experiment_name}/{model_name}/{bias_name}_seed-{i}.mat", \
            mdict={'x_train': x_AM_train, 'x_dev': x_AM_dev, \
                'y_p_train': one_hot_biases_train, 'y_p_dev': one_hot_biases_dev, \
                'y_m_train': y_m_train, 'y_m_dev': y_m_dev, \
                'y_m_train_one_hot': y_m_train_one_hot, 'y_m_dev_one_hot': y_m_dev_one_hot})

        A = np.dot(x_AM_train.T, one_hot_biases_train) / x_AM_train.shape[0]
        U_experiment, s, vh = np.linalg.svd(A, full_matrices=True)

        io.savemat(f"data/projection_matrix/{experiment_name}/{model_name}/SAL_{bias_name}_partialSup_seed-{i}.mat", \
                mdict={'best_iter_X': x_AM_train, 'best_iter_Z': one_hot_biases_train, 'best_iter_Y': y_m_train, 'U_experiment':U_experiment})




    # experiment_name = "twitter-labelled-by-overlap-short"
    # x_AM_train = twitter_short_train
    # x_AM_dev = twitter_short_dev
    # x_tpr_train = twitter_short_train
    # x_tpr_dev = twitter_short_dev

    # np.savez(f"data/{experiment_name}/{model_name}/{bias_name}.npz", x_train = x_tpr_train, x_dev = x_tpr_dev, \
    #     y_p_train = biases_train, y_p_dev = biases_dev, \
    #     y_m_train = y_m_train, y_m_dev = y_m_dev, \
    #     y_m_train_one_hot = y_m_train_one_hot, y_m_dev_one_hot = y_m_dev_one_hot)

    # # biases_train, biases_dev = one_hot_to_1d(one_hot_biases_train, one_hot_biases_dev)

    # for i in range(10):
    #     one_hot_biases_train = np.concatenate((one_hot_age_train, one_hot_gender_train[i, :]), axis=1)
    #     one_hot_biases_dev = np.concatenate((one_hot_age_dev, one_hot_gender_dev[i, :]), axis=1)

    #     io.savemat(f"data/{experiment_name}/{model_name}/{bias_name}_seed-{i}.mat", \
    #         mdict={'x_train': x_AM_train, 'x_dev': x_AM_dev, \
    #             'y_p_train': one_hot_biases_train, 'y_p_dev': one_hot_biases_dev, \
    #             'y_m_train': y_m_train, 'y_m_dev': y_m_dev, \
    #             'y_m_train_one_hot': y_m_train_one_hot, 'y_m_dev_one_hot': y_m_dev_one_hot})

    #     A = np.dot(x_AM_train.T, one_hot_biases_train) / x_AM_train.shape[0]
    #     U_experiment, s, vh = np.linalg.svd(A, full_matrices=True)

    #     io.savemat(f"data/projection_matrix/{experiment_name}/{model_name}/SAL_{bias_name}_partialSup_seed-{i}.mat", \
    #               mdict={'best_iter_X': x_AM_train, 'best_iter_Z': one_hot_biases_train, 'best_iter_Y': y_m_train, 'U_experiment':U_experiment})


    end = time.time()
    print(end - start)