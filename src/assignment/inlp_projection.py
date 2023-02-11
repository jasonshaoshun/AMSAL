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
    "--experiment",
    action="store",
    type=str,
    help="Experiment name",
)

parser.add_argument(
    "--assignment",
    action="store",
    type=str,
    choices=["Kmeans", "Oracle", "Sal", "partialSup"],
    help="The assignment used",
)

parser.add_argument(
    "--bias",
    action="store",
    type=str,
    help="The bias exhibits in the dataset",
)

parser.add_argument(
    "--dataset_path",
    action="store",
    type=str,
    help="dataset path",
)

parser.add_argument(
    "--assignment_path",
    action="store",
    type=str,
    help="assignment path",
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


def load_inlp_assigned_dataset(dataset_path, assignment_path, testset=True):

    saved_dataset = scipy.io.loadmat(assignment_path)
    n_x = saved_dataset['best_iter_X'].shape[0]
    n_train = np.floor(n_x * 0.7).astype(int)

    x_train = saved_dataset['best_iter_X']
    y_p_train_assignment = saved_dataset['best_iter_Z']
    y_m_train = saved_dataset['best_iter_Y']

    # y_p_train = y_p_train[:, 0]
    y_p_train_assignment = y_p_train_assignment[:, 0]
    # y_p_train = y_p_train.reshape((-1, ))
    y_p_train_assignment = y_p_train_assignment.reshape((-1, ))
    y_m_train = y_m_train.reshape((-1, ))
    
    saved_dataset = np.load(dataset_path)
    x_dev = saved_dataset['x_dev']
    y_p_dev = saved_dataset['y_p_dev']
    y_m_dev = saved_dataset['y_m_dev']




    if testset == True:
        x_test = saved_dataset['x_test']
        y_p_test = saved_dataset['y_p_test']
        y_m_test = saved_dataset['y_m_test']

        return x_train, y_p_train_assignment, y_m_train, x_dev, y_p_dev, y_m_dev, x_test, y_p_test, y_m_test

    else:
        return x_train, y_p_train_assignment, y_m_train, x_dev, y_p_dev, y_m_dev


def load_assigned_dataset(assignment_mat_path, dataset_npz_path):
    
    saved_dataset = scipy.io.loadmat(assignment_mat_path)

    x_train = saved_dataset['best_iter_X']
    y_m_train = saved_dataset['best_iter_Y']
    print(f"y_m_train {y_m_train.shape}")
    y_m_train = y_m_train.reshape((-1, ))
    y_p_train = saved_dataset['best_iter_Z']
    print(f"y_p_train shape is {y_p_train.shape}, y_p_train[1:10, :] {y_p_train[1:10, :]}\n\n\n")
    y_p_train = map(lambda num: 1 if (num==[1, 0]).all() else 0, y_p_train)
    y_p_train = np.asarray(list(y_p_train))
    y_p_train = y_p_train.reshape((-1, ))

    saved_dataset = np.load(f"{dataset_npz_path}")
    x_dev = saved_dataset['x_dev']
    y_m_dev = saved_dataset['y_m_dev']
    print(f"y_m_dev {y_m_dev.shape}")
    y_m_dev = y_m_dev.reshape((-1, ))
    y_p_dev = saved_dataset['y_p_dev']
    print(f"y_p_dev {y_p_dev.shape}")
    y_p_dev = y_p_dev.reshape((-1, ))

    print(f"x_train {x_train.shape}, y_p_train {y_p_train.shape}, y_m_train {y_m_train.shape}, x_dev {x_dev.shape}, y_p_dev {y_p_dev.shape}, y_m_dev {y_m_dev.shape}")

    return x_train, y_p_train, y_m_train, x_dev, y_p_dev, y_m_dev



def compute_biography_projection(assignment_path, dataset_path):

    x_train, y_p_train, y_m_train, x_dev, y_p_dev, y_m_dev\
                = load_assigned_dataset(assignment_path, dataset_path)

    is_autoregressive = True
    min_acc = 0.
    if "BertModel"  in assignment_path:
        dim = 768
        n = 300
    elif "FastText" in assignment_path:
        dim = 300
        n = 150

    TYPE= "svm"
    gender_clf = LinearSVC
    params = {'penalty': 'l2', 'C': 0.01, 'fit_intercept': True, 'class_weight': None, "dual": False}

    P, rowspace_projections, Ws = debias.get_debiasing_projection(gender_clf, params, n, dim, is_autoregressive, min_acc,
                                        x_train, y_p_train, x_dev, y_p_dev,
                                        Y_train_main=y_m_train, Y_dev_main=y_m_dev, by_class = True)
    return P

def compute_deepmoji_projection(assignment_path, dataset_path):
    x_train, y_p_train, y_m_train, x_dev, y_p_dev, y_m_dev \
                = load_assigned_dataset(assignment_path, dataset_path)
    
    results = defaultdict(dict)
    n = 300
    dim = 300
    is_autoregressive = True
    min_acc = 0.51
    noise = False

    clf = LogisticRegression
    params = {'penalty': 'l1', 'C': 0.1, 'solver': 'saga'}
    # clf = LinearSVC
    # {'fit_intercept': True, 'class_weight': 'balanced', 'dual': False, 'C': 0.1}    

    P_n = debias.get_debiasing_projection(clf, params, n, dim, is_autoregressive, min_acc,
                                            x_train, y_p_train, x_dev, y_p_dev,
                                            by_class=True, Y_train_main=y_m_train, Y_dev_main=y_m_dev)

    P = P_n[1]
    n_dims = 200
    P = debias.get_projection_to_intersection_of_nullspaces(P[:n_dims], input_dim=300)

    return P


if __name__ == "__main__":
    args = parser.parse_args()

    # dataset_to_dataset_folder = {
    #     "Bert": "/data/assignment/old/fair_biography_prof_gender/BERT/all.npz",
    #     "Fasttext": "/data/assignment/old/fair_biography_prof_gender/FastText/all.npz",
    #     "Deepmoji05": "/data/assignment/old/fair_emoji_sent_race/0.5/all.npz",
    #     "Deepmoji08": "/data/assignment/old/fair_emoji_sent_race/0.8/all.npz",
    #     "Deepmoji80": "/data/assignment/old/fair_emoji_sent_race/80/all.npz"
    # }

    # dataset_to_eigenvector_folder = {
    #     "Bert": "data/assignment/projection_matrix/BERT/",
    #     "Fasttext": "data/assignment/projection_matrix/FastText/",
    #     "Deepmoji05": "data/assignment/projection_matrix/05_all/",
    #     "Deepmoji08": "data/assignment/projection_matrix/08_all/",
    #     "Deepmoji80": "data/assignment/projection_matrix/80_all/"
    # }


    if args.experiment == "biography" or args.experiment == "biography-labelled-by-overlap" or args.experiment == "biography-different-partial-n":
        P = compute_biography_projection(args.assignment_path, args.dataset_path)
    elif args.experiment == "deepmoji":
        P = compute_deepmoji_projection(args.assignment_path, args.dataset_path)


    # np.savez(f"{args.save_path}/INLP_{args.bias}_{args.assignment}.npz", P=P)
    if args.supervision_ratio == "null":
        if args.seed == "null":
            np.savez(f"{args.save_path}/INLP_{args.bias}_{args.assignment}.npz", P=P)
        else:
            np.savez(f"{args.save_path}/INLP_{args.bias}_{args.assignment}_seed-{args.seed}.npz", P=P)
    else:
        if args.seed == "null":
            np.savez(f"{args.save_path}/INLP_{args.bias}_{args.assignment}_np-{args.supervision_ratio}.npz", P=P)