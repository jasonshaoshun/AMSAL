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


def load_word_vectors(fname):
    
    model = KeyedVectors.load_word2vec_format(fname, binary=False)
    # vecs = model.vectors
    # words = list(model.index_to_key)
    # return model, vecs, words
    return model


if __name__ == "__main__":
    args = parser.parse_args()
    start = time.time()

    # k = 10

    model_name = "FastText"
    # device = "cpu"
    # device = "cuda"
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
    gender_group = np.asarray(data["gender_group"])


    ######################################################################################
    # Encode Y1: Age
    one_hot_age = map(lambda num: [1, 0, 0] if (num==0) else ([0, 1, 0] if (num==1) else [0, 0, 1]), age_group)
    one_hot_age = np.asarray(list(one_hot_age))
    one_hot_age_train = one_hot_age[:train_n, :]
    one_hot_age_dev = one_hot_age[train_n:, :]

    ######################################################################################
    # Encode Y2: Gender
    one_hot_gender = map(lambda num: [0, 1] if (num==0) else [1, 0], gender_group)
    one_hot_gender = np.asarray(list(one_hot_gender))
    one_hot_gender_train = one_hot_gender[:train_n, :]
    one_hot_gender_dev = one_hot_gender[train_n:, :]

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





    # # # Save the dataset by location_concat only

    # # bias_name = "location_concat"
    # # one_hot_biases = location_concat


    # # # Save the dataset by location only

    # # bias_name = "location"
    # # one_hot_biases = location


    # # # Save the dataset by sentiment only

    # # bias_name = "sentiment"
    # # one_hot_biases = sentiment_one_hot

    # # # Save the dataset by age, gender and location

    # # bias_name = "age_gender_location"
    # # one_hot_biases = np.concatenate((one_hot_age, one_hot_gender, location), axis=1)








    # Few things to change before starting whole experiments:
    # 1.keep eyes on x_train, it should be =Tweet on normal twitter and =db_text on concat twitter.



    experiment_name = f"twitter-concat-sentiment-{args.version}"
    x_AM_train = twitter_concat_train
    x_AM_dev = twitter_concat_dev
    x_tpr_train = twitter_short_train
    x_tpr_dev = twitter_short_dev

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





    # Save the dataset by age only

    bias_name = "age"
    one_hot_biases_train = one_hot_age_train
    one_hot_biases_dev = one_hot_age_dev
    biases_train, biases_dev = one_hot_to_1d(one_hot_biases_train, one_hot_biases_dev)

    io.savemat(f"data/{experiment_name}/{model_name}/{bias_name}.mat", \
        mdict={'x_train': x_AM_train, 'x_dev': x_AM_dev, \
            'y_p_train': one_hot_biases_train, 'y_p_dev': one_hot_biases_dev, \
            'y_m_train': y_m_train, 'y_m_dev': y_m_dev, \
            'y_m_train_one_hot': y_m_train_one_hot, 'y_m_dev_one_hot': y_m_dev_one_hot})

    np.savez(f"data/{experiment_name}/{model_name}/{bias_name}.npz", x_train = x_tpr_train, x_dev = x_tpr_dev, \
            y_p_train = biases_train, y_p_dev = biases_dev, \
            y_m_train = y_m_train, y_m_dev = y_m_dev, \
            y_m_train_one_hot = y_m_train_one_hot, y_m_dev_one_hot = y_m_dev_one_hot)


    # Save the dataset by gender only

    bias_name = "gender"
    one_hot_biases_train = one_hot_gender_train
    one_hot_biases_dev = one_hot_gender_dev
    biases_train, biases_dev = one_hot_to_1d(one_hot_biases_train, one_hot_biases_dev)



    io.savemat(f"data/{experiment_name}/{model_name}/{bias_name}.mat", \
        mdict={'x_train': x_AM_train, 'x_dev': x_AM_dev, \
            'y_p_train': one_hot_biases_train, 'y_p_dev': one_hot_biases_dev, \
            'y_m_train': y_m_train, 'y_m_dev': y_m_dev, \
            'y_m_train_one_hot': y_m_train_one_hot, 'y_m_dev_one_hot': y_m_dev_one_hot})
            
    np.savez(f"data/{experiment_name}/{model_name}/{bias_name}.npz", x_train = x_tpr_train, x_dev = x_tpr_dev, \
            y_p_train = biases_train, y_p_dev = biases_dev, \
            y_m_train = y_m_train, y_m_dev = y_m_dev, \
            y_m_train_one_hot = y_m_train_one_hot, y_m_dev_one_hot = y_m_dev_one_hot)


    # Save the dataset by age and gender

    bias_name = "age-gender"
    one_hot_biases_train = np.concatenate((one_hot_age_train, one_hot_gender_train), axis=1)
    one_hot_biases_dev = np.concatenate((one_hot_age_dev, one_hot_gender_dev), axis=1)
    biases_train, biases_dev = one_hot_to_1d(one_hot_biases_train, one_hot_biases_dev)



    io.savemat(f"data/{experiment_name}/{model_name}/{bias_name}.mat", \
        mdict={'x_train': x_AM_train, 'x_dev': x_AM_dev, \
            'y_p_train': one_hot_biases_train, 'y_p_dev': one_hot_biases_dev, \
            'y_m_train': y_m_train, 'y_m_dev': y_m_dev, \
            'y_m_train_one_hot': y_m_train_one_hot, 'y_m_dev_one_hot': y_m_dev_one_hot})
            
    np.savez(f"data/{experiment_name}/{model_name}/{bias_name}.npz", x_train = x_tpr_train, x_dev = x_tpr_dev, \
            y_p_train = biases_train, y_p_dev = biases_dev, \
            y_m_train = y_m_train, y_m_dev = y_m_dev, \
            y_m_train_one_hot = y_m_train_one_hot, y_m_dev_one_hot = y_m_dev_one_hot)








    # experiment_name = f"twitter-short-sentiment-{args.version}"
    # x_AM_train = twitter_short_train
    # x_AM_dev = twitter_short_dev
    # x_tpr_train = twitter_short_train
    # x_tpr_dev = twitter_short_dev

    # y_m_train = sentiment_train
    # y_m_dev = sentiment_dev
    # y_m_train_one_hot = sentiment_one_hot_train
    # y_m_dev_one_hot = sentiment_one_hot_dev

    # # Save the dataset by age only

    # bias_name = "age"
    # one_hot_biases_train = one_hot_age_train
    # one_hot_biases_dev = one_hot_age_dev
    # biases_train, biases_dev = one_hot_to_1d(one_hot_biases_train, one_hot_biases_dev)



    # io.savemat(f"data/{experiment_name}/{model_name}/{bias_name}.mat", \
    #        mdict={'x_train': x_AM_train, 'x_dev': x_AM_dev, \
    #         'y_p_train': one_hot_biases_train, 'y_p_dev': one_hot_biases_dev, \
    #         'y_m_train': y_m_train, 'y_m_dev': y_m_dev, \
    #         'y_m_train_one_hot': y_m_train_one_hot, 'y_m_dev_one_hot': y_m_dev_one_hot})
    # np.savez(f"data/{experiment_name}/{model_name}/{bias_name}.npz", x_train = x_tpr_train, x_dev = x_tpr_dev, \
    #         y_p_train = biases_train, y_p_dev = biases_dev, \
    #         y_m_train = y_m_train, y_m_dev = y_m_dev, \
    #         y_m_train_one_hot = y_m_train_one_hot, y_m_dev_one_hot = y_m_dev_one_hot)


    # # Save the dataset by gender only

    # bias_name = "gender"
    # one_hot_biases_train = one_hot_gender_train
    # one_hot_biases_dev = one_hot_gender_dev
    # biases_train, biases_dev = one_hot_to_1d(one_hot_biases_train, one_hot_biases_dev)



    # io.savemat(f"data/{experiment_name}/{model_name}/{bias_name}.mat", \
    #        mdict={'x_train': x_AM_train, 'x_dev': x_AM_dev, \
    #         'y_p_train': one_hot_biases_train, 'y_p_dev': one_hot_biases_dev, \
    #         'y_m_train': y_m_train, 'y_m_dev': y_m_dev, \
    #         'y_m_train_one_hot': y_m_train_one_hot, 'y_m_dev_one_hot': y_m_dev_one_hot})
    # np.savez(f"data/{experiment_name}/{model_name}/{bias_name}.npz", x_train = x_tpr_train, x_dev = x_tpr_dev, \
    #         y_p_train = biases_train, y_p_dev = biases_dev, \
    #         y_m_train = y_m_train, y_m_dev = y_m_dev, \
    #         y_m_train_one_hot = y_m_train_one_hot, y_m_dev_one_hot = y_m_dev_one_hot)


    # # Save the dataset by age and gender

    # bias_name = "age-gender"
    # one_hot_biases_train = np.concatenate((one_hot_age_train, one_hot_gender_train), axis=1)
    # one_hot_biases_dev = np.concatenate((one_hot_age_dev, one_hot_gender_dev), axis=1)
    # biases_train, biases_dev = one_hot_to_1d(one_hot_biases_train, one_hot_biases_dev)



    # io.savemat(f"data/{experiment_name}/{model_name}/{bias_name}.mat", \
    #        mdict={'x_train': x_AM_train, 'x_dev': x_AM_dev, \
    #         'y_p_train': one_hot_biases_train, 'y_p_dev': one_hot_biases_dev, \
    #         'y_m_train': y_m_train, 'y_m_dev': y_m_dev, \
    #         'y_m_train_one_hot': y_m_train_one_hot, 'y_m_dev_one_hot': y_m_dev_one_hot})
    # np.savez(f"data/{experiment_name}/{model_name}/{bias_name}.npz", x_train = x_tpr_train, x_dev = x_tpr_dev, \
    #         y_p_train = biases_train, y_p_dev = biases_dev, \
    #         y_m_train = y_m_train, y_m_dev = y_m_dev, \
    #         y_m_train_one_hot = y_m_train_one_hot, y_m_dev_one_hot = y_m_dev_one_hot)






    end = time.time()
    print(end - start)


