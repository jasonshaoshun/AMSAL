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
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    "--bias",
    action="store",
    type=str,
    help="biases to be removed e.g. age and gender",
)

parser.add_argument(
    "--debiasing",
    action="store",
    type=str,
    choices=["SAL", "INLP"],
    help="The debiasing method",
)

parser.add_argument(
    "--experiment",
    action="store",
    type=str,
    help="The experiment name",
)

parser.add_argument(
    "--assignment",
    action="store",
    type=str,
    help="The assignment used",
)

parser.add_argument(
    "--dataset_path",
    action="store",
    type=str,
    help="The path to the dataset stored in npz",
)

parser.add_argument(
    "--projection_matrix_file",
    action="store",
    type=str,
    help="The file store INLP and SAL projection matrix",
)

parser.add_argument(
    "--projection_matrix_path",
    action="store",
    type=str,
    help="The path to the projection matrix of INLP and SAL stored in npz/mat",
)

parser.add_argument(
    "--save_path",
    action="store",
    type=str,
    help="The path to the store the model",
)


parser.add_argument(
    "--num_eigenvectors_to_remove",
    action="store",
    type=int,
    help="number of singular vectors to remove by SAL",
)

parser.add_argument(
    "--supervision_ratio",
    action="store",
    type=str,
    default="null",
    help="The assignment used",
)

parser.add_argument(
    "--seed",
    action="store",
    type=str,
    default="null",
    help="The seed on the word list",
)

parser.add_argument(
    "--dataset_version",
    action="store",
    type=str,
    default="null",
    help="The version of dataset",
)




def map_one_hot_2d_to_1d(y):

    y = map(lambda num: 0 if (num==[1, 0]).all() else 1, y)
    y = np.asarray(list(y))

    return y

def map_one_hot_3d_to_1d(y):

    y = map(lambda num: 0 if (num==[1, 0, 0]).all() else (1 if (num==[0, 1, 0]).all() else 2), y)
    y = np.asarray(list(y))

    return y


def load_dataset(dataset_path):

    saved_dataset = np.load(f"{dataset_path}")

    x_train = saved_dataset['x_train']
    y_m_train = saved_dataset['y_m_train']
    y_m_train = y_m_train.reshape((-1, ))
    y_p_train = saved_dataset['y_p_train']
    y_p_train = y_p_train.reshape((-1, ))

    x_dev = saved_dataset['x_dev']
    y_m_dev = saved_dataset['y_m_dev']
    y_m_dev = y_m_dev.reshape((-1, ))
    y_p_dev = saved_dataset['y_p_dev']
    y_p_dev = y_p_dev.reshape((-1, ))

    return x_train, y_p_train, y_m_train, x_dev, y_p_dev, y_m_dev





def load_eigenvectors(data_path, debiasing_method):
    if debiasing_method == "SAL":
        saved_model = scipy.io.loadmat(f"{data_path}.mat")
        U_experiment = saved_model['U_experiment']
        
    elif debiasing_method == "INLP":
        saved_model = np.load(f"{data_path}.npz")
        U_experiment = saved_model['P']

    return U_experiment


def load_true_Z_and_assignment_Z(data_path):
    
    
    saved_model = scipy.io.loadmat(data_path)
    
    y_p_gold = saved_model['best_iter_original_Z']
    y_p_learnt = saved_model['best_iter_Z']
    
    return y_p_gold, y_p_learnt


def debiase_X(num_eigenvectors_to_remove, U, x_train, x_test, debiasing_method):
    if debiasing_method == "SAL":
        u_r = U[:, num_eigenvectors_to_remove:]
        proj = u_r @ u_r.T
        P = proj
    elif debiasing_method == "INLP":
        P = U

    debiased_x_train = P.dot(x_train.T).T
    debiased_x_test = P.dot(x_test.T).T
    return debiased_x_train, debiased_x_test


# def get_TPR_unweighted(y_pred, y_true, protected_attribute, tpr_metric):
#     print("============================ unweighted ============================")
#     if protected_attribute.shape[1] == 2:
#         protected_attribute = map_one_hot_2d_to_1d(protected_attribute)
#     elif protected_attribute.shape[1] == 3:
#         protected_attribute = map_one_hot_3d_to_1d(protected_attribute)
    
#     # protected_attribute_dim = np.unique(protected_attribute).shape[0]
    
#     scores = defaultdict(Counter)
#     sent_count_total = defaultdict(Counter)
    
#     y_pred = y_pred.reshape((-1, ))
#     y_true = y_true.reshape((-1, ))
#     protected_attribute = protected_attribute.reshape((-1, ))
#     for y_hat, y, g in zip(y_pred, y_true, protected_attribute):
#         if y == y_hat:
#             scores[y][g] += 1
#         sent_count_total[y][g] += 1

#     values, counts = np.unique(y_pred, return_counts=True)
#     distribution_y_hat_main = zip(values, counts)
#     print(f"\n\ndistribution_y_hat_main {list(distribution_y_hat_main)}\n")
    
#     values, counts = np.unique(y_true, return_counts=True)
#     distribution_y_main = zip(values, counts)
#     print(f"distribution_y_main {list(distribution_y_main)}\n")


#     print(f"\ntrue count(y_hat==y_true): scores[sentiment][biases] {scores},\n\ntotal count: count[sentiment][biases] {sent_count_total}\n")

#     print(f"\ntpr_metric {tpr_metric}\n")

#     tprs = defaultdict(dict)
#     tprs_change = dict()

#     for sentiment, scores_dict in scores.items():        
#         for protect_attribute_class in sent_count_total[sentiment]:
#             tprs[sentiment][protect_attribute_class] = scores_dict[protect_attribute_class] / sent_count_total[sentiment][protect_attribute_class]

#         if tpr_metric == 'gap':
#             tprs_on_same_sentiment = np.asarray(list(tprs[sentiment].values()))
#             tprs_change[sentiment] = (tprs_on_same_sentiment[0] - tprs_on_same_sentiment[1]) ** 2
#         elif tpr_metric == 'variance':
#             tprs_change[sentiment] = np.var(np.asarray(list(tprs[sentiment].values())))
#         else:
#             sys.exit("tpr_metric is not found")

#         print(f"\nsentiment {sentiment}, tprs[sentiment] {tprs[sentiment]},\ntprs_change[sentiment] {tprs_change[sentiment]}")


#         # if len(sent_count_total[sentiment]) == protected_attribute_dim:

#         # else:    
#         #     print(f"sent_count_total[sentiment] {sent_count_total[sentiment]}\n\
#         #     sentiment {sentiment} missed in tpr calculation\n")


#     if tpr_metric == 'gap':
#         tprs_change = np.sqrt(np.mean(np.asarray(list(tprs_change.values()))))
#     elif tpr_metric == 'variance':
#         tprs_change = np.sqrt(np.mean(np.array(list(tprs_change.values()))))

#     print(f"tprs_change is {tprs_change}\n\n\n")

#     return tprs, tprs_change




# def get_TPR(y_pred, y_true, protected_attribute, tpr_metric, y_true_train):

#     print("============================ weighted ============================")

#     if protected_attribute.shape[1] == 2:
#         protected_attribute = map_one_hot_2d_to_1d(protected_attribute)
#     elif protected_attribute.shape[1] == 3:
#         protected_attribute = map_one_hot_3d_to_1d(protected_attribute)
    
#     # protected_attribute_dim = np.unique(protected_attribute).shape[0]
    
#     scores = defaultdict(Counter)
#     sent_count_total = defaultdict(Counter)
    
#     y_pred = y_pred.reshape((-1, ))
#     y_true = y_true.reshape((-1, ))
#     protected_attribute = protected_attribute.reshape((-1, ))
#     for y_hat, y, g in zip(y_pred, y_true, protected_attribute):
#         if y == y_hat:
#             scores[y][g] += 1
#         sent_count_total[y][g] += 1

#     values, counts = np.unique(y_pred, return_counts=True)
#     distribution_y_hat_main = zip(values, counts)
#     print(f"\n\ndistribution_y_hat_main {list(distribution_y_hat_main)}\n")
    
#     values, counts = np.unique(y_true, return_counts=True)
#     distribution_y_main = zip(values, counts)
#     print(f"distribution_y_main {list(distribution_y_main)}\n")
    

#     print(f"\ntrue count(y_hat==y_true): scores[sentiment][biases] {scores},\n\ntotal count: count[sentiment][biases] {sent_count_total}\n")

#     print(f"\ntpr_metric {tpr_metric}\n")


#     tprs = defaultdict(dict)
#     tprs_change = dict()
#     weight_array = np.array([])
#     for sentiment, scores_dict in scores.items():
        
#         for protect_attribute_class in sent_count_total[sentiment]:
#             tprs[sentiment][protect_attribute_class] = scores_dict[protect_attribute_class] / sent_count_total[sentiment][protect_attribute_class]
#             # print(f"\n\nsentiment is {sentiment}, protect_attribute_class is {protect_attribute_class}, tprs[sentiment] is {tprs[sentiment]}, \
#             # scores_dict[protect_attribute_class] is {scores_dict[protect_attribute_class]}, \
#             # sent_count_total[sentiment][protect_attribute_class] is {sent_count_total[sentiment][protect_attribute_class]}\n\n")
        

#         weight = (y_true_train == sentiment).mean()
#         weight_array = np.append(weight_array, weight)

#         if tpr_metric == 'gap':
#             tprs_on_same_sentiment = np.asarray(list(tprs[sentiment].values()))
#             tprs_change[sentiment] = weight * ((tprs_on_same_sentiment[0] - tprs_on_same_sentiment[1]) ** 2)
#         elif tpr_metric == 'variance':
#             tprs_change[sentiment] = weight * np.var(np.asarray(list(tprs[sentiment].values())))
#         else:
#             sys.exit("tpr_metric is not found")

#         print(f"\nsentiment {sentiment}, weight {weight}, tprs[sentiment] {tprs[sentiment]}, tprs_change[sentiment] {tprs_change[sentiment]}")


#     # if len(sent_count_total[sentiment]) == protected_attribute_dim:

#     # else:    
#     #     print(f"sent_count_total[sentiment] {sent_count_total[sentiment]}\n\
#     #     sentiment {sentiment} missed in tpr calculation\n")

#     print(f"\nweight_array is {weight_array}\n")
    
#     # if weight_array.sum() != 1:
#     #     print("weight is not summed up to 1, please check")
#     #     exit()


#     if tpr_metric == 'gap':
#         tprs_change = np.sqrt(np.sum(np.asarray(list(tprs_change.values()))))
#     elif tpr_metric == 'variance':
#         tprs_change = np.sqrt(np.sum(np.array(list(tprs_change.values()))))

#     print(f"tprs_change is {tprs_change}\n\n\n")

#     return tprs, tprs_change



def msemaegap(y_pred, y_true, z, mseormae):
#     print(f"y_pred shape {y_pred.shape}, y_true shape {y_true.shape}, z shape is {z.shape}")
    C = np.unique(z, axis=0)
    gap = np.array([])
    
    for unique_value in C:
        index = np.all(z == unique_value, axis=1)
        unique_value_y_true = y_true[index]
        unique_value_y_pred = y_pred[index]
        if mseormae == 'mse':
            errs = np.square(unique_value_y_true - unique_value_y_pred)
            my_mean = np.mean(errs)
            tprgap_like = np.mean(np.square(errs - my_mean))

            # print(f"unique_value_y_true - unique_value_y_pred is \n{unique_value_y_true - unique_value_y_pred}\n\n")
            # print(f"np.square(unique_value_y_true - unique_value_y_pred) is \n{np.square(unique_value_y_true - unique_value_y_pred)}\n\n")
            # print(f"my_mean {np.mean(np.square(unique_value_y_true - unique_value_y_pred))}")
            # print(f"unsorted: errs - my_mean is \n {errs - my_mean}")
            # print(f"sorted: errs - my_mean  is \n {np.sort(errs - my_mean)}")
            # print(f"tprgap_like {tprgap_like}\n\n\n\n\n")

        elif mseormae == 'mae':
            errs = np.abs(unique_value_y_true - unique_value_y_pred)
            my_mean = np.mean(errs)
            tprgap_like = np.mean(np.abs(errs - my_mean))
        else:
            print("error")
    
        gap = np.append(gap, tprgap_like)
    
    # Version 1.0
    # gap_sum = np.sum(gap)
    # gap_mean = gap_sum / gap.shape[0]
    
    # if mseormae == 'mse':
    #     gap_mean = np.sqrt(gap_mean)

    # Version 2.0
    gap_std = np.std(gap)

    print(f"gap is {gap}, gap_std is {gap_std}")

    return gap_std







def run_political_twitter_experiments(model, dataset_path, projection_matrix_file, num_eigenvectors_to_remove, debiasing_method, dataset_version):

    x_train, y_p_train, y_m_train, x_test, y_p_test, y_m_test = load_dataset(dataset_path)
    U_experiment = load_eigenvectors(projection_matrix_file, debiasing_method)

    y_p_test = map(lambda num: [1, 0, 0, 1, 0] if (num==1).all() else ([1, 0, 0, 0, 1] if (num==2).all() \
    else ([0, 1, 0, 1, 0] if (num==3).all() else([0, 1, 0, 0, 1] if (num==4).all() \
    else ([0, 0, 1, 1, 0] if (num==5).all() else [0, 0, 1, 0, 1])))), y_p_test)
    y_p_test = np.asarray(list(y_p_test))
    print("=========================================== Start of BIASED experiments ===========================================")

    random.seed(0)
    np.random.seed(0)


    if model == "BertModel":
        if dataset_version == "v3":
            clf_original = LogisticRegression(penalty = "l2", C = 1.0, random_state = 0, solver = "saga", max_iter = 1000)
        elif dataset_version == "v4":
            clf_original = LinearRegression(fit_intercept = False, positive = True)
        else:
            print("dataset version is unknown")
            exit()
    elif model == "FastText":
        if dataset_version == "v3":
            clf_original = LogisticRegression(penalty = "l1", C = 0.1, random_state = 0, solver = "liblinear", max_iter = 1000)
        elif dataset_version == "v4":
            clf_original = LinearRegression(fit_intercept = True, positive = False)
        else:
            print("dataset version is unknown")
            exit()


    clf_original.fit(x_train, y_m_train)
    biased_score = clf_original.score(x_test, y_m_test)

    y_pred_before = clf_original.predict(x_test)
    mean_squared_error_before = mean_squared_error(y_m_test, y_pred_before)
    mean_absolute_error_before = mean_absolute_error(y_m_test, y_pred_before)

    print(f"Score of profession classifier on original(biased) dataset {biased_score}, mean_squared_error_before {mean_squared_error_before}, mean_absolute_error_before {mean_absolute_error_before}\n")

    print(f"\ny_m_test max is {np.max(y_m_test)}\ny_m_test is {y_m_test}\n\ny_pred_before max is {np.max(y_pred_before)}\ny_pred_before is {y_pred_before}\n\n")

    # f1_macro_before = f1_score(y_m_test, y_pred_before.astype(int), labels=None, average='macro')
    # f1_micro_before = f1_score(y_m_test, y_pred_before.astype(int), labels=None, average='micro')
    # tprs_before_age, tprs_change_before_age = get_TPR(y_pred_before.astype(int), y_m_test, y_p_test[:, 0:3], 'variance', y_m_train)
    # tprs_before_gender, tprs_change_before_gender = get_TPR(y_pred_before.astype(int), y_m_test, y_p_test[:, 3:5], 'gap', y_m_train)
    # _, tprs_change_before_age_unweighted = get_TPR_unweighted(y_pred_before.astype(int), y_m_test, y_p_test[:, 0:3], 'variance')
    # _, tprs_change_before_gender_unweighted = get_TPR_unweighted(y_pred_before.astype(int), y_m_test, y_p_test[:, 3:5], 'gap')
    # tpr_gap_before = [tprs_change_before_age, tprs_change_before_gender]
    # tpr_gap_before_unweighted = [tprs_change_before_age_unweighted, tprs_change_before_gender_unweighted]

    MSE_age_before = msemaegap(y_pred_before, y_m_test, y_p_test[:, 0:3], 'mse')
    MAE_age_before = msemaegap(y_pred_before, y_m_test, y_p_test[:, 0:3], 'mae')
    MSE_gender_before = msemaegap(y_pred_before, y_m_test, y_p_test[:, 3:5], 'mse')
    MAE_gender_before = msemaegap(y_pred_before, y_m_test, y_p_test[:, 3:5], 'mae')


    print("=========================================== Start of DeBIASED experiments ===========================================")

    debiased_x_train, debiased_x_test = debiase_X(num_eigenvectors_to_remove, U_experiment, x_train, x_test, debiasing_method)

    random.seed(0)
    np.random.seed(0)

    if model == "BertModel":
        if dataset_version == "v3":
            clf_original = LogisticRegression(penalty = "l2", C = 1.0, random_state = 0, solver = "saga", max_iter = 1000)
        elif dataset_version == "v4":
            clf_original = LinearRegression(fit_intercept = False, positive = True)
        else:
            print("dataset version is unknown")
            exit()
    elif model == "FastText":
        if dataset_version == "v3":
            clf_original = LogisticRegression(penalty = "l1", C = 0.1, random_state = 0, solver = "liblinear", max_iter = 1000)
        elif dataset_version == "v4":
            clf_original = LinearRegression(fit_intercept = True, positive = False)
        else:
            print("dataset version is unknown")
            exit()

    clf_original.fit(debiased_x_train, y_m_train)

    debiased_score = clf_original.score(debiased_x_test, y_m_test)
    y_pred_after = clf_original.predict(debiased_x_test)
    mean_squared_error_after = mean_squared_error(y_m_test, y_pred_after)
    mean_absolute_error_after = mean_absolute_error(y_m_test, y_pred_after)

    print(f"Score of profession classifier on original(biased) dataset {debiased_score}, mean_squared_error_after {mean_squared_error_after}, mean_absolute_error_after {mean_absolute_error_after}\n")

    print(f"\ny_m_test max is {np.max(y_m_test)}\ny_m_test is {y_m_test}\n\ny_pred_after max is {np.max(y_pred_after)}\ny_pred_after is {y_pred_after}\n\n")

    # tprs_after_age, tprs_change_after_age = get_TPR(y_pred_after.astype(int), y_m_test, y_p_test[:, 0:3], 'variance', y_m_train)
    # tprs_after_gender, tprs_change_after_gender = get_TPR(y_pred_after.astype(int), y_m_test, y_p_test[:, 3:5], 'gap', y_m_train)
    # _, tprs_change_after_age_unweighted = get_TPR_unweighted(y_pred_after.astype(int), y_m_test, y_p_test[:, 0:3], 'variance')
    # _, tprs_change_after_gender_unweighted = get_TPR_unweighted(y_pred_after.astype(int), y_m_test, y_p_test[:, 3:5], 'gap')
    # tpr_gap_after = [tprs_change_after_age, tprs_change_after_gender]
    # tpr_gap_after_unweighted = [tprs_change_after_age_unweighted, tprs_change_after_gender_unweighted]
    # f1_macro_after = f1_score(y_m_test, y_pred_after.astype(int), labels=None, average='macro')
    # f1_micro_after = f1_score(y_m_test, y_pred_after.astype(int), labels=None, average='micro')

    MSE_age_after = msemaegap(y_pred_after, y_m_test, y_p_test[:, 0:3], 'mse')
    MAE_age_after = msemaegap(y_pred_after, y_m_test, y_p_test[:, 0:3], 'mae')
    MSE_gender_after = msemaegap(y_pred_after, y_m_test, y_p_test[:, 3:5], 'mse')
    MAE_gender_after = msemaegap(y_pred_after, y_m_test, y_p_test[:, 3:5], 'mae')

    print(f"mean_squared_error_before {mean_squared_error_before}, mean_absolute_error_before {mean_absolute_error_before}, mean_squared_error_after {mean_squared_error_after}, mean_absolute_error_after {mean_absolute_error_after}\n")
    print(f"MSE_age_after {MSE_age_after}, MAE_age_after {MAE_age_after}, MSE_gender_after {MSE_gender_after}, MAE_gender_after {MAE_gender_after}\n")

    return mean_squared_error_before, mean_absolute_error_before, mean_squared_error_after, mean_absolute_error_after, MSE_age_before, MAE_age_before, MSE_gender_before, MAE_gender_before, MSE_age_after, MAE_age_after, MSE_gender_after, MAE_gender_after




def generate_experiment_id(
    name,
    debiasing=None,
    model=None,
    experiment=None,
    assignment=None,
    bias=None,
    seed=None,
    supervision_ratio=None,
):
    experiment_id = f"{name}"

    # Build the experiment ID.
    if isinstance(debiasing, str):
        experiment_id += f"_d-{debiasing}"
    if isinstance(model, str):
        experiment_id += f"_m-{model}"
    if isinstance(experiment, str):
        experiment_id += f"_e-{experiment}"
    if isinstance(assignment, str):
        experiment_id += f"_a-{assignment}"
    if isinstance(bias, str):
        experiment_id += f"_b-{bias}"
    if isinstance(seed, str):
        if seed != "null":
            experiment_id += f"_s-{seed}"
    if isinstance(supervision_ratio, str):
        if supervision_ratio != "null":
            experiment_id += f"_r-{supervision_ratio}"
    
    return experiment_id

if __name__ == "__main__":
    args = parser.parse_args()

    # The experiment_id can be used as the file name to be saved
    experiment_id = generate_experiment_id(
        name="TPRGAP",
        debiasing=args.debiasing,
        model=args.model,
        experiment=args.experiment,
        assignment=args.assignment,
        bias=args.bias,
        seed=args.seed,
        supervision_ratio=args.supervision_ratio,
    )
    
    print(f"supervision_ratio is {args.supervision_ratio}, experiment_id is {experiment_id}")
    
    if args.experiment == "twitter-different-partial-n-concat" or args.experiment == "twitter-different-partial-n-short" or args.experiment == "twitter-different-partial-n-concat-v4":
        y_p_gold, y_p_learnt = load_true_Z_and_assignment_Z(f"{args.projection_matrix_path}/SAL_{args.bias}_{args.assignment}_np-{args.supervision_ratio}.mat")
        assignment_accuracy = np.sum(np.all(y_p_gold == y_p_learnt, axis=1)/y_p_gold.shape[0])
    else:
        assignment_accuracy=0
    
    print(f"args.num_eigenvectors_to_remove is {args.num_eigenvectors_to_remove}")
    mean_squared_error_before, mean_absolute_error_before, mean_squared_error_after, mean_absolute_error_after, \
    MSE_age_before, MAE_age_before, MSE_gender_before, MAE_gender_before, \
    MSE_age_after, MAE_age_after, MSE_gender_after, MAE_gender_after = \
    run_political_twitter_experiments(args.model, args.dataset_path, args.projection_matrix_file, args.num_eigenvectors_to_remove, args.debiasing, args.dataset_version)

    print(f"assignment_accuracy is {assignment_accuracy},\n\
    mean_squared_error_before is {mean_squared_error_before},\n\
    mean_absolute_error_before is {mean_absolute_error_before},\n\
    mean_squared_error_after is {mean_squared_error_after},\n\
    mean_absolute_error_after is {mean_absolute_error_after},\n\
    MSE_age_before is {MSE_age_before},\n\
    MAE_age_before is {MAE_age_before},\n\
    MSE_gender_before is {MSE_gender_before},\n\
    MAE_gender_before is {MAE_gender_before},\n\
    MSE_age_after is {MSE_age_after},\n\
    MAE_age_after is {MAE_age_after},\n\
    MSE_gender_after is {MSE_gender_after},\n\
    MAE_gender_after is {MAE_gender_after}")

    result = []
    result.append(
        {
            "experiment_id": experiment_id,
            "assignment_accuracy": assignment_accuracy,
            "mean_squared_error_before": mean_squared_error_before,
            "mean_absolute_error_before": mean_absolute_error_before,
            "mean_squared_error_after": mean_squared_error_after,
            "mean_absolute_error_after": mean_absolute_error_after,
            "MSE_age_before": MSE_age_before,
            "MAE_age_before": MAE_age_before,
            "MSE_gender_before": MSE_gender_before,
            "MAE_gender_before": MAE_gender_before, 
            "MSE_age_after": MSE_age_after, 
            "MAE_age_after": MAE_age_after, 
            "MSE_gender_after": MSE_gender_after,
            "MAE_gender_after": MAE_gender_after,
        }
    )

    # result = []
    # result.append({"biased_accuracy": biased_accuracy, "debiased_accuracy_by_models": debiased_accuracy_by_models,})

    # os.makedirs(f"{args.persistent_dir}/src/assignment/results/political_twitter/tpr-gap", exist_ok=True)
    # with open(f"{args.persistent_dir}/src/assignment/results/political_twitter/tpr-gap/{experiment_id}.json", "w") as f:
    #     json.dump(result, f)
        
    os.makedirs(f"{args.save_path}", exist_ok=True)
    with open(f"{args.save_path}/{experiment_id}.json", "w") as f:
        json.dump(result, f)
