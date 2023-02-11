import sys
sys.path.append("../CCA_debiase/")
sys.path.append("../CCA_debiase/src")
import debias
import numpy as np
from sklearn.linear_model import LogisticRegression
import random
from collections import defaultdict, Counter
from typing import List
import matplotlib.pyplot as plt
import scipy.io
from scipy.stats.stats import pearsonr
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
import os
import argparse
import json

parser = argparse.ArgumentParser(description="Run the tpr-gap experiments in INLP")

parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    help="Directory where all persistent data will be stored.",
)

parser.add_argument(
    "--model",
    action="store",
    type=str,
    help="Models used to encode the context, e.g. BertModel",
)

parser.add_argument(
    "--assignment",
    action="store",
    type=str,
    choices=["partialSup", "Kmeans", "Oracle", "Sal"],
    help="AM modes",
)

parser.add_argument(
    "--bias",
    action="store",
    type=str,
    help="The bias exhibits in the dataset",
)

parser.add_argument(
    "--assignment_path",
    action="store",
    type=str,
    help="The path of assignment file learnt in AM",
)


parser.add_argument(
    "--dataset_path",
    action="store",
    type=str,
    help="The path of assignment file learnt in AM",
)


parser.add_argument(
    "--save_path",
    action="store",
    type=str,
    help="The path of to save INLP projection matrix",
)

parser.add_argument(
    "--supervision_ratio",
    action="store",
    type=str,
    default="null",
    help="The ratio of samples has been used when find the best iterations in AM",
)

parser.add_argument(
    "--seed",
    action="store",
    type=str,
    default="null",
    help="The dataset seed",
)


parser.add_argument(
    "--dataset_version",
    action="store",
    type=str,
    default="null",
    help="The version of dataset",
)


def load_inlp_assigned_dataset(assignment_mat_path, dataset_npz_path):

    saved_dataset = scipy.io.loadmat(f"{assignment_mat_path}")

    x_train = saved_dataset['best_iter_X']
    y_m_train = saved_dataset['best_iter_Y']
    y_m_train = y_m_train.reshape((-1, ))
    y_p_train = saved_dataset['best_iter_Z']
    y_p_train = map(lambda num: 1 if (num==[1, 0, 0, 1, 0]).all() else (2 if (num==[1, 0, 0, 0, 1]).all() \
    else (3 if (num==[0, 1, 0, 1, 0]).all() else(4 if (num==[0, 1, 0, 0, 1]).all() \
    else (5 if (num==[0, 0, 1, 1, 0]).all() else 6)))), y_p_train)
    # y_p_train = map(lambda num: 1 if (num==[1, 0, 0, 1, 0]).all() else (2 if (num==[1, 0, 0, 0, 1]).all() else (3 if (num==[0, 1, 0, 1, 0]).all() else(4 if (num==[0, 1, 0, 0, 1]).all() else (5 if (num==[0, 0, 1, 1, 0]).all() else 6)))), y_p_train)
    y_p_train = np.asarray(list(y_p_train))
    y_p_train = y_p_train.reshape((-1, ))

    saved_dataset = np.load(f"{dataset_npz_path}")
    x_dev = saved_dataset['x_dev']
    y_m_dev = saved_dataset['y_m_dev']
    y_m_dev = y_m_dev.reshape((-1, ))
    y_p_dev = saved_dataset['y_p_dev']
    y_p_dev = y_p_dev.reshape((-1, ))

    # print(f"x_train {x_train.shape}, y_p_train {y_p_train.shape}, y_m_train {y_m_train.shape}, x_dev {x_dev.shape}, y_p_dev {y_p_dev.shape}, y_m_dev {y_m_dev.shape}")
    
    return x_train, y_p_train, y_m_train, x_dev, y_p_dev, y_m_dev

# def load_inlp_assigned_dataset(assignment_path, testset=True):

#     saved_dataset = scipy.io.loadmat(f"{assignment_path}")
#     n_x = saved_dataset['best_iter_X'].shape[0]
#     n_train = np.floor(n_x * 0.7).astype(int)

#     x_train = saved_dataset['best_iter_X'][:n_train, :]
#     y_m_train = saved_dataset['best_iter_Y'][:n_train, :]
#     y_m_train = y_m_train.reshape((-1, ))
#     y_p_train = saved_dataset['best_iter_Z'][:n_train, :]
#     y_p_train = map(lambda num: 1 if (num==[1, 0, 0, 1, 0]).all() else (2 if (num==[1, 0, 0, 0, 1]).all() \
#     else (3 if (num==[0, 1, 0, 1, 0]).all() else(4 if (num==[0, 1, 0, 0, 1]).all() \
#     else (5 if (num==[0, 0, 1, 1, 0]).all() else 6)))), y_p_train)
#     # y_p_train = map(lambda num: 1 if (num==[1, 0, 0, 1, 0]).all() else (2 if (num==[1, 0, 0, 0, 1]).all() else (3 if (num==[0, 1, 0, 1, 0]).all() else(4 if (num==[0, 1, 0, 0, 1]).all() else (5 if (num==[0, 0, 1, 1, 0]).all() else 6)))), y_p_train)
#     y_p_train = np.asarray(list(y_p_train))

#     x_dev = saved_dataset['best_iter_X'][n_train:, :]
#     y_m_dev = saved_dataset['best_iter_Y'][n_train:, :]
#     y_m_dev = y_m_dev.reshape((-1, ))
#     y_p_dev = saved_dataset['best_iter_Z'][n_train:, :]
#     y_p_dev = map(lambda num: 1 if (num==[1, 0, 0, 1, 0]).all() else (2 if (num==[1, 0, 0, 0, 1]).all() \
#     else (3 if (num==[0, 1, 0, 1, 0]).all() else(4 if (num==[0, 1, 0, 0, 1]).all() \
#     else (5 if (num==[0, 0, 1, 1, 0]).all() else 6)))), y_p_dev)
#     y_p_dev = np.asarray(list(y_p_dev))
    

#     if testset == True:
#         x_test = saved_dataset['x_test']
#         y_p_test = saved_dataset['y_p_test']
#         y_m_test = saved_dataset['y_m_test']

#         return x_train, y_p_train, y_m_train, x_dev, y_p_dev, y_m_dev, x_test, y_p_test, y_m_test

#     else:
#         return x_train, y_p_train, y_m_train, x_dev, y_p_dev, y_m_dev



def compute_twitter_political_projection(model, assignment_mat_path, dataset_npz_path, dataset_version):
    x_train, y_p_train, y_m_train, x_dev, y_p_dev, y_m_dev \
                = load_inlp_assigned_dataset(assignment_mat_path, dataset_npz_path)
    
    is_autoregressive = True
    min_acc = 0.
    # max_iter = 100

    if model == "BertModel":
        dim = 768
        if dataset_version == "v3":
            params = {'penalty': 'l1', 'C': 0.1, 'solver': 'liblinear'}
        elif dataset_version == "v4":
            params = {'penalty': 'elasticnet', 'C': 0.1, 'solver': 'saga', 'l1_ratio': 0.5}
        else:
            print("dataset version is unknown")
            exit()
    elif model == "FastText":
        dim = 300
        if dataset_version == "v3":
            params = {'penalty': 'l1', 'C': 0.1, 'solver': 'liblinear'}
        elif dataset_version == "v4":
            params = {'penalty': 'l1', 'C': 0.1, 'solver': 'liblinear'}
        else:
            print("dataset version is unknown")
            exit()
    
    n = 300
    gender_clf = LogisticRegression
    
    print(f"x_train {x_train.shape},y_p_train {y_p_train.shape},y_m_train {y_m_train.shape},x_dev {x_dev.shape},y_p_dev {y_p_dev.shape},y_m_dev {y_m_dev.shape}")
    P, rowspace_projections, Ws = debias.get_debiasing_projection(gender_clf, params, n, dim, is_autoregressive, min_acc,
                                        x_train, y_p_train, x_dev, y_p_dev,
                                        Y_train_main=y_m_train, Y_dev_main=y_m_dev, by_class = True)
    return P


if __name__ == "__main__":
    args = parser.parse_args()

    P = compute_twitter_political_projection(args.model, args.assignment_path, args.dataset_path, args.dataset_version)

    if args.supervision_ratio == "null":
        if args.seed == "null":
            np.savez(f"{args.save_path}/INLP_{args.bias}_{args.assignment}.npz", P=P)
        else:
            np.savez(f"{args.save_path}/INLP_{args.bias}_{args.assignment}_seed-{args.seed}.npz", P=P)
    else:
        if args.seed == "null":
            np.savez(f"{args.save_path}/INLP_{args.bias}_{args.assignment}_np-{args.supervision_ratio}.npz", P=P)

