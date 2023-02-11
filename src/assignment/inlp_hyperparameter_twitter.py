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
import warnings
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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

# parser.add_argument(
#     "--assignment",
#     action="store",
#     type=str,
#     choices=["partialSup", "Kmeans", "Oracle", "Sal"],
#     help="AM modes",
# )

# parser.add_argument(
#     "--bias",
#     action="store",
#     type=str,
#     help="The bias exhibits in the dataset",
# )

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







def load_majority_dataset(dataset_npz_path):

    saved_dataset = np.load(f"{dataset_npz_path}")
    x_train = saved_dataset['x_train']
    y_m_train = saved_dataset['y_majority_train']
    print(f"y_m_train {y_m_train.shape}")
    y_m_train = y_m_train.reshape((-1, ))
    y_p_train = saved_dataset['y_p_train']
    print(f"y_p_train {y_p_train.shape}")
    y_p_train = y_p_train.reshape((-1, ))

    x_dev = saved_dataset['x_dev']
    y_m_dev = saved_dataset['y_majority_dev']
    print(f"y_m_dev {y_m_dev.shape}")
    y_m_dev = y_m_dev.reshape((-1, ))
    y_p_dev = saved_dataset['y_p_dev']
    print(f"y_p_dev {y_p_dev.shape}")
    y_p_dev = y_p_dev.reshape((-1, ))

    print(f"x_train {x_train.shape}, y_p_train {y_p_train.shape}, y_m_train {y_m_train.shape}, x_dev {x_dev.shape}, y_p_dev {y_p_dev.shape}, y_m_dev {y_m_dev.shape}")

    return x_train, y_p_train, y_m_train, x_dev, y_p_dev, y_m_dev





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




def find_best_hyperparameters_logistic_regression(assignment_mat_path, dataset_npz_path, classify_main_or_protected = "main_label"):
    x_train, y_p_train, y_m_train, x_dev, y_p_dev, y_m_dev \
                = load_inlp_assigned_dataset(assignment_mat_path, dataset_npz_path)
    best_acc = 0
    max_iter = 1000
    
    hyperparameter_performance = []
    
    if classify_main_or_protected == "protected_label":
        y_train = y_p_train
        y_dev = y_p_dev
    elif classify_main_or_protected == "main_label":
        y_train = y_m_train
        y_dev = y_m_dev

    for penalty in ['l1', 'l2', 'elasticnet']:
        for C in [0.1, 0.5, 1.0, 10.0]:
            for solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
                
                if solver == 'newton-cg' and (penalty != 'l2' and penalty != 'none'):
                    continue
                if solver == 'lbfgs' and (penalty != 'l2' and penalty != 'none'):
                    continue
                if solver == 'liblinear' and (penalty != 'l1' and penalty != 'l2'):
                    continue
                if solver == 'sag' and (penalty != 'l2' and penalty != 'none'):
                    continue

                if penalty == 'elasticnet':
                    l1_ratio = 0.5
                else:
                    l1_ratio = None

                print(f"\n\n\npenalty {penalty}, C {C}, solver {solver}\n\n\n")
                clf_original = LogisticRegression(penalty = penalty, C = C, random_state = 0, solver = solver, 
                                                  warm_start = False, multi_class = 'auto', fit_intercept = True, 
                                                  verbose = 10, n_jobs = None, max_iter = max_iter, l1_ratio = l1_ratio)
                # with warnings.catch_warnings():
                #     warnings.filterwarnings('error')
                #     try:
                #         clf_original.fit(x_train, y_train)
                #     except:
                #         print(f"ConvergenceWarning: penalty {penalty}, C {C}, solver {solver}")
                #         continue
                clf_original.fit(x_train, y_train)
                
                biased_accuracy = clf_original.score(x_dev, y_dev)
                performance = {'penalty': penalty, 'C': C, 'solver': solver, 'biased_accuracy': biased_accuracy}
                hyperparameter_performance.append(performance)
                print(performance)
                if biased_accuracy > best_acc:
                    print(f"Update, biased_accuracy is {biased_accuracy}")
                    best_acc = biased_accuracy
                    best_penalty = penalty
                    best_C = C
                    best_solver = solver
    if best_acc == 0:
        best_acc = 0
        best_penalty = 0
        best_C = 0
        best_solver = 0

    return best_acc, best_penalty, best_C, best_solver, hyperparameter_performance


def find_best_hyperparameters_linearSVC(assignment_mat_path, dataset_npz_path, classify_main_or_protected = "main_label"):
    x_train, y_p_train, y_m_train, x_dev, y_p_dev, y_m_dev \
                = load_inlp_assigned_dataset(assignment_mat_path, dataset_npz_path)
    best_acc = 0
    max_iter = 10000
    penalty = 'l2'
    hyperparameter_performance = []
    
    if classify_main_or_protected == "protected_label":
        y_train = y_p_train
        y_dev = y_p_dev
    elif classify_main_or_protected == "main_label":
        y_train = y_m_train
        y_dev = y_m_dev


    for loss in ['hinge', 'squared_hinge']:
        for C in [0.1, 0.5, 1.0, 10.0]:
            for multi_class in ['ovr', 'crammer_singer']:
                for class_weight in [None, 'balanced']:
                    
                    if penalty == 'l1' and (loss == 'hinge' or loss == 'squared_hinge'):
                        continue

                    print(f"\n\n\npenalty {penalty}, loss {loss}, C {C}, multi_class {multi_class}, class_weight {class_weight}\n\n\n")
                    clf_original = LinearSVC(penalty = penalty, loss = loss, C = C, multi_class = multi_class, class_weight = class_weight, random_state = 0, max_iter = max_iter)

                    # with warnings.catch_warnings():
                    #     warnings.filterwarnings('error')
                    #     try:
                    #         clf_original.fit(x_train, y_train)
                    #     except Warning:
                    #         print(f"ConvergenceWarning: penalty {penalty}, loss {loss}, C {C}, multi_class {multi_class}, class_weight {class_weight}")
                    #         continue
                    clf_original.fit(x_train, y_train)


                    biased_accuracy = clf_original.score(x_dev, y_dev)
                    performance = {'penalty': penalty, 'loss': loss, 'C': C, 'multi_class': multi_class, 'class_weight': class_weight, 'biased_accuracy': biased_accuracy}
                    hyperparameter_performance.append(performance)
                    print(performance)
                    if biased_accuracy > best_acc:
                        print(f"Update, biased_accuracy is {biased_accuracy}")
                        best_acc = biased_accuracy
                        best_penalty = penalty
                        best_loss = loss
                        best_C = C
                        best_multi_class = multi_class
                        best_class_weight = class_weight
    if best_acc == 0:
        best_acc = 0
        best_penalty = 0
        best_loss = 0
        best_C = 0
        best_multi_class = 0
        best_class_weight = 0
    
    return best_acc, best_penalty, best_loss, best_C, best_multi_class, best_class_weight, hyperparameter_performance








def find_best_hyperparameters_linear_regression(dataset_npz_path, classify_main_or_protected = "main_label"):
    
    x_train, y_p_train, y_m_train, x_dev, y_p_dev, y_m_dev = load_majority_dataset(dataset_npz_path)
         
    best_squared_error = 1000000
    max_iter = 1000
    
    hyperparameter_performance = []
    
    if classify_main_or_protected == "protected_label":
        y_train = y_p_train
        y_dev = y_p_dev
    elif classify_main_or_protected == "main_label":
        y_train = y_m_train
        y_dev = y_m_dev

    for fit_intercept in [True, False]:
        for positive in [True, False]:


            print(f"\n\n\nfit_intercept {fit_intercept}, positive {positive}\n\n\n")
            clf_original = LinearRegression(fit_intercept = fit_intercept, positive = positive)
            clf_original.fit(x_train, y_train)

            y_hat = clf_original.predict(x_dev)
            squared_error = mean_squared_error(y_dev, y_hat)
            
            performance = {'fit_intercept': fit_intercept, 'positive': positive, 'squared_error': squared_error}
            hyperparameter_performance.append(performance)
            print(performance)
            
            if squared_error < best_squared_error:
                print(f"Update, squared_error is {squared_error}")
                best_squared_error = squared_error
                best_fit_intercept = fit_intercept
                best_positive = positive
                
    if best_squared_error == 0:
        best_fit_intercept = 0
        best_positive = 0

    return best_squared_error, best_fit_intercept, best_positive, hyperparameter_performance



if __name__ == "__main__":

    args = parser.parse_args()


    # print(f"start experiment 1: on logistic regression, protected label")
    # best_acc, best_penalty, best_C, best_solver, hyperparameter_performance = find_best_hyperparameters_logistic_regression(args.assignment_path, args.dataset_path, classify_main_or_protected = "protected_label")

    # result = []
    # result.append(
    #     {
    #         "best_acc": best_acc,
    #         "best_penalty": best_penalty,
    #         "best_C": best_C,
    #         "best_solver": best_solver,
    #         "hyperparameter_performance": hyperparameter_performance,
    #     }
    # )

    # with open(f"{args.save_path}/best_hyperparameter_logistic-regression_protected-label.json", "w") as f:
    #     json.dump(result, f)



    # print(f"start experiment 2: on logistic regression, main label")   
    # best_acc, best_penalty, best_C, best_solver, hyperparameter_performance = find_best_hyperparameters_logistic_regression(args.assignment_path, args.dataset_path, classify_main_or_protected = "main_label")
    
    # result = []
    # result.append(
    #     {
    #         "best_acc": best_acc,
    #         "best_penalty": best_penalty,
    #         "best_C": best_C,
    #         "best_solver": best_solver,
    #         "hyperparameter_performance": hyperparameter_performance,
    #     }
    # )

    # with open(f"{args.save_path}/best_hyperparameter_logistic-regression_main-label.json", "w") as f:
    #     json.dump(result, f)




    # print(f"start experiment 3: on linearSVC, protected label")
    # best_acc, best_penalty, best_loss, best_C, best_multi_class, best_class_weight, hyperparameter_performance = find_best_hyperparameters_linearSVC(args.assignment_path, args.dataset_path, classify_main_or_protected = "protected_label")

    # result = []
    # result.append(
    #     {
    #         "best_acc": best_acc,
    #         "best_penalty": best_penalty,
    #         "best_loss": best_loss,
    #         "best_C": best_C,
    #         "best_multi_class": best_multi_class,
    #         "best_class_weight": best_class_weight,
    #         "hyperparameter_performance": hyperparameter_performance,
    #     }
    # )

    # with open(f"{args.save_path}/best_hyperparameter_linearSVC_protected-label.json", "w") as f:
    #     json.dump(result, f)




    # print(f"start experiment 4: on linearSVC, main label")
    # best_acc, best_penalty, best_loss, best_C, best_multi_class, best_class_weight, hyperparameter_performance = find_best_hyperparameters_linearSVC(args.assignment_path, args.dataset_path, classify_main_or_protected = "main_label")
    
    # result = []
    # result.append(
    #     {
    #         "best_acc": best_acc,
    #         "best_penalty": best_penalty,
    #         "best_loss": best_loss,
    #         "best_C": best_C,
    #         "best_multi_class": best_multi_class,
    #         "best_class_weight": best_class_weight,
    #         "hyperparameter_performance": hyperparameter_performance,
    #     }
    # )

    # with open(f"{args.save_path}/best_hyperparameter_linearSVC_main-label.json", "w") as f:
    #     json.dump(result, f)





























    # print(f"start experiment 1: on linear regression, protected label")
    # best_squared_error, best_fit_intercept, best_positive, hyperparameter_performance = find_best_hyperparameters_linear_regression(args.dataset_path, classify_main_or_protected = "protected_label")

    # result = []
    # result.append(
    #     {
    #         "best_squared_error": best_squared_error,
    #         "best_fit_intercept": best_fit_intercept,
    #         "best_positive": best_positive,
    #         "hyperparameter_performance": hyperparameter_performance,
    #     }
    # )

    # with open(f"{args.save_path}/best_hyperparameter_linear-regression_protected-label.json", "w") as f:
    #     json.dump(result, f)



    print(f"start experiment 2: on linear regression, main label")   
    best_squared_error, best_fit_intercept, best_positive, hyperparameter_performance = find_best_hyperparameters_linear_regression(args.dataset_path, classify_main_or_protected = "main_label")
    
    result = []
    result.append(
        {
            "best_squared_error": best_squared_error,
            "best_fit_intercept": best_fit_intercept,
            "best_positive": best_positive,
            "hyperparameter_performance": hyperparameter_performance,
        }
    )

    with open(f"{args.save_path}/best_hyperparameter_linear-regression_main-label.json", "w") as f:
        json.dump(result, f)


