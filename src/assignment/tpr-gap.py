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
import sys
from sklearn.metrics import f1_score

np.set_printoptions(threshold=sys.maxsize)
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
    choices=["Kmeans", "Oracle", "Sal", "partialSup"],
    help="The assignment used",
)

parser.add_argument(
    "--dataset_path",
    action="store",
    type=str,
    help="dataset path",
)

parser.add_argument(
    "--projection_matrix_file",
    action="store",
    type=str,
    help="assignment path",
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
    help="save path",
)

# parser.add_argument(
#     "--variable_with_changing_ratio",
#     action="store",
#     type=str,
#     help="assignment path",
# )

parser.add_argument(
    "--train_ratio",
    action="store",
    type=str,
    help="ratio of the training",
)

parser.add_argument(
    "--test_ratio",
    action="store",
    type=str,
    help="ratio of the test",
)

parser.add_argument(
    "--assignment_ratio",
    action="store",
    type=str,
    help="ratio of the test",
)


parser.add_argument(
    "--eigenvectors_to_remove",
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

def load_dictionary(path):
    
    with open(path, "r", encoding = "utf-8") as f:
        
        lines = f.readlines()
        
    k2v, v2k = {}, {}
    for line in lines:
        
        k,v = line.strip().split("\t")
        v = int(v)
        k2v[k] = v
        v2k[v] = k
    
    return k2v, v2k


def get_TPR_unweighted(y_pred, y_true, p2i, i2p, y_p_test):
    gender = np.where(y_p_test == 1, 'f', 'm')
    scores = defaultdict(Counter)
    prof_count_total = defaultdict(Counter)
    
    for y_hat, y, g in zip(y_pred, y_true, gender):
        
        if y == y_hat:
            
            scores[i2p[y]][g] += 1
        
        prof_count_total[i2p[y]][g] += 1
    

    values, counts = np.unique(y_pred, return_counts=True)
    distribution_y_hat_main = zip(values, counts)
    print(f"\n\ndistribution_y_hat_main {list(distribution_y_hat_main)}\n")
    
    values, counts = np.unique(y_true, return_counts=True)
    distribution_y_main = zip(values, counts)
    print(f"distribution_y_main {list(distribution_y_main)}\n")
    
    print(f"\nprof_count_total {prof_count_total}\n")


    tprs = defaultdict(dict)
    tprs_change = dict()
    tprs_ratio = []

    for profession, scores_dict in scores.items():
        
        good_m, good_f = scores_dict["m"], scores_dict["f"]
        prof_total_f = prof_count_total[profession]["f"]
        prof_total_m = prof_count_total[profession]["m"]
        if prof_total_m > 0 and prof_total_f > 0:
            tpr_m = (good_m) / prof_total_m
            tpr_f = (good_f) / prof_total_f

            tprs[profession]["m"] = tpr_m
            tprs[profession]["f"] = tpr_f
            tprs_ratio.append(0)

            tprs_change[profession] = (tpr_f - tpr_m) ** 2
            print(f"profession is {profession}, tpr male {tprs[profession]['m']}, tpr female {tprs[profession]['f']}.\ntprs_change[profession] {tprs_change[profession]}")

        else:
            print(f"profession {profession} missed in tpr-calc:\n\
                  number of {profession} male in test set {prof_total_m}\n\
                  number of {profession} female in test set {prof_total_f}\n\n")


    tpr_gap = np.sqrt(np.mean(np.array(list(tprs_change.values()))))

    print(f"\ntprs_change {list(tprs_change.values())}, \ntpr_gap is {tpr_gap}\n\n")

    return tprs, tprs_change, np.mean(np.abs(tprs_ratio)), tpr_gap


def get_TPR(y_pred, y_true, p2i, i2p, y_p_test):
    gender = np.where(y_p_test == 1, 'f', 'm')
    scores = defaultdict(Counter)
    prof_count_total = defaultdict(Counter)
    
    for y_hat, y, g in zip(y_pred, y_true, gender):
        
        if y == y_hat:
            
            scores[i2p[y]][g] += 1
        
        prof_count_total[i2p[y]][g] += 1
    

    values, counts = np.unique(y_pred, return_counts=True)
    distribution_y_hat_main = zip(values, counts)
    print(f"\n\ndistribution_y_hat_main {list(distribution_y_hat_main)}\n")
    
    values, counts = np.unique(y_true, return_counts=True)
    distribution_y_main = zip(values, counts)
    print(f"distribution_y_main {list(distribution_y_main)}\n")
    
    print(f"\nprof_count_total {prof_count_total}\n")


    tprs = defaultdict(dict)
    tprs_change = dict()
    tprs_ratio = []
    weight_array = np.array([])
    for profession, scores_dict in scores.items():
        
        good_m, good_f = scores_dict["m"], scores_dict["f"]
        prof_total_f = prof_count_total[profession]["f"]
        prof_total_m = prof_count_total[profession]["m"]
        if prof_total_m > 0 and prof_total_f > 0:
            tpr_m = (good_m) / prof_total_m
            tpr_f = (good_f) / prof_total_f

            tprs[profession]["m"] = tpr_m
            tprs[profession]["f"] = tpr_f
            tprs_ratio.append(0)

            # print(f"y_p_test shape {y_p_test.shape}, p2i[profession] is {p2i[profession]}")
            weight = (y_true == p2i[profession]).mean()
            weight_array = np.append(weight_array, weight)

            tprs_change[profession] = weight * ((tpr_f - tpr_m) ** 2)


            print(f"profession is {profession}, weight {weight}, tpr male {tprs[profession]['m']}, tpr female {tprs[profession]['f']}\ntprs_change[profession] {tprs_change[profession]}")

        else:
            print(f"profession {profession} missed in tpr-calc:\n\
                  number of {profession} male in test set {prof_total_m}\n\
                  number of {profession} female in test set {prof_total_f}\n\n")
    
    if 1 - weight_array.sum() > 0.001:
        print(f"weight_array is {weight_array}, {weight_array.sum()}, weight is not summed up to 1, please check")
        exit()

    tpr_gap = np.sqrt(np.sum(np.array(list((tprs_change.values())))))

    print(f"\ntprs_change {list(tprs_change.values())}, \ntpr_gap is {tpr_gap}\n\n")

    return tprs, tprs_change, np.mean(np.abs(tprs_ratio)), tpr_gap
    
def similarity_vs_tpr(tprs, word2vec, title, measure, prof2fem):
    
    professions = list(tprs.keys())
    #
    """ 
    sims = dict()
    gender_direction = word2vec["he"] - word2vec["she"]
    
    for p in professions:
        sim = word2vec.cosine_similarities(word2vec[p], [gender_direction])[0]
        sims[p] = sim
    """
    tpr_lst = [tprs[p] for p in professions]
    sim_lst = [prof2fem[p] for p in professions]

    #professions = [p.replace("_", " ") for p in professions if p in word2vec]
    
    plt.plot(sim_lst, tpr_lst, marker = "o", linestyle = "none")
    plt.xlabel("% women", fontsize = 13)
    plt.ylabel(r'$GAP_{female,y}^{TPR}$', fontsize = 13)
    for p in professions:
        x,y = prof2fem[p], tprs[p]
        plt.annotate(p , (x,y), size = 7, color = "red")
    plt.ylim(-0.4, 0.55)
    z = np.polyfit(sim_lst, tpr_lst, 1)
    p = np.poly1d(z)
    plt.plot(sim_lst,p(sim_lst),"r--")
    plt.savefig("{}_vs_bias_{}_bert".format(measure, title), dpi = 600)
    print("Correlation: {}; p-value: {}".format(*pearsonr(sim_lst, tpr_lst)))
    plt.show()


def get_TPR_emoji(y_main, y_hat_main, y_protected, y_m_train):
    
    all_y = list(Counter(y_main).keys())
    
    protected_vals = defaultdict(dict)
    dict_format_y = defaultdict(dict)
    for label in all_y:
        for i in range(2):
            used_vals = (y_main == label) & (y_protected == i)
            y_label = y_main[used_vals]
            y_hat_label = y_hat_main[used_vals]
            protected_vals['y:{}'.format(label)]['p:{}'.format(i)] = (y_label == y_hat_label).mean()
            dict_format_y['y:{}'.format(label)] = label

    diffs = {}
    weight_array = np.array([])
    
    values, counts = np.unique(y_hat_main, return_counts=True)
    distribution_y_hat_main = zip(values, counts)
    print(f"\n\ndistribution_y_hat_main {list(distribution_y_hat_main)}\n")
    
    values, counts = np.unique(y_main, return_counts=True)
    distribution_y_main = zip(values, counts)
    print(f"distribution_y_main {list(distribution_y_main)}\n")
    

    print(f"\nprotected_vals {protected_vals}, dict_format_y {dict_format_y}\n")
    
    for k, v in protected_vals.items():
        vals = list(v.values())
        weight = (y_m_train == dict_format_y[k]).mean()
        diffs[k] = weight * ((vals[0] - vals[1]) ** 2)
        print(f"\n{k}, weight is {weight}, v is {v}, diffs {diffs[k]}")
        weight_array = np.append(weight_array, weight)
    print(f"\nsum of diffs is {np.sum(list(diffs.values()))}, weight_array is {weight_array}\n\n\n\n\n")
    if weight_array.sum() != 1:
        print("weight is not summed up to 1, please check")
        exit()
    
    tpr_gap = np.sqrt(np.sum((list(diffs.values()))))
    
    return protected_vals, tpr_gap


def get_TPR_emoji_unweighted(y_main, y_hat_main, y_protected, y_m_train):
    
    all_y = list(Counter(y_main).keys())
    
    protected_vals = defaultdict(dict)
    dict_format_y = defaultdict(dict)
    for label in all_y:
        for i in range(2):
            used_vals = (y_main == label) & (y_protected == i)
            y_label = y_main[used_vals]
            y_hat_label = y_hat_main[used_vals]
            protected_vals['y:{}'.format(label)]['p:{}'.format(i)] = (y_label == y_hat_label).mean()
            dict_format_y['y:{}'.format(label)] = label

    diffs = {}
    
    values, counts = np.unique(y_hat_main, return_counts=True)
    distribution_y_hat_main = zip(values, counts)
    print(f"\n\ndistribution_y_hat_main {list(distribution_y_hat_main)}\n")
    
    values, counts = np.unique(y_main, return_counts=True)
    distribution_y_main = zip(values, counts)
    print(f"distribution_y_main {list(distribution_y_main)}\n")
    

    print(f"\nprotected_vals {protected_vals}, dict_format_y {dict_format_y}\n")
    
    for k, v in protected_vals.items():
        vals = list(v.values())
        diffs[k] = (vals[0] - vals[1]) ** 2
        print(f"\n{k}, v is {v}, diffs {diffs[k]}")

    print(f"\nmean of diffs is {np.mean(list(diffs.values()))}")
    tpr_gap = np.sqrt(np.mean((list(diffs.values()))))
        
    return protected_vals, tpr_gap



def load_true_Z_and_assignment_Z(data_path):
    
    
    saved_model = scipy.io.loadmat(data_path)
    
    y_p_gold = saved_model['best_iter_original_Z']
    y_p_learnt = saved_model['best_iter_Z']
    
    return y_p_gold, y_p_learnt


# def rms_diff(tpr_diff):
#     return np.sqrt(np.mean(tpr_diff**2))

def rms(arr):
    # return np.sqrt(np.mean(np.square(arr)))
    return np.sqrt(np.sum(arr))


def load_dataset(data_path):
    saved_dataset = np.load(f"{data_path}")

    x_train = saved_dataset['x_train']
    y_m_train = saved_dataset['y_m_train']
    y_p_train = saved_dataset['y_p_train']

    x_dev = saved_dataset['x_dev']
    y_p_dev = saved_dataset['y_p_dev']
    y_m_dev = saved_dataset['y_m_dev']

    x_test = saved_dataset['x_test']
    y_p_test = saved_dataset['y_p_test']
    y_m_test = saved_dataset['y_m_test']
        
    return x_train, y_m_train, y_p_train, x_dev, y_p_dev, y_m_dev, x_test, y_p_test, y_m_test

def load_deep_moji_dataset(data_path, ratio, target_set):
    
    saved_dataset = np.load(f"{data_path}/{ratio}/all.npz")
    if target_set == "train":
        x = saved_dataset['x_train']
        y_m = saved_dataset['y_m_train']
        y_p = saved_dataset['y_p_train']
    elif target_set == "test":
        x = saved_dataset['x_dev']
        y_m = saved_dataset['y_m_dev']
        y_p = saved_dataset['y_p_dev']

    return x, y_m, y_p

def load_eigenvectors(data_path, debiasing_method):
    if debiasing_method == "SAL":
        saved_model = scipy.io.loadmat(data_path)
        U_experiment = saved_model['U_experiment']
        # U_experiment = saved_model['U_best']
    elif debiasing_method == "INLP":
        saved_model = np.load(f"{data_path}.npz")
        U_experiment = saved_model['P']

    return U_experiment

def run_deepmoji_experiments(dataset_path, eigenvector_path, eigenvectors_to_remove, debiasing_method, train_ratio, test_ratio):

    x_train, y_m_train, y_p_train = load_deep_moji_dataset(dataset_path, train_ratio, "train")
    x_dev, y_m_dev, y_p_dev = load_deep_moji_dataset(dataset_path, test_ratio, "test")

#     print(f"dataset_path {train_ratio}, test_ratio {test_ratio}, y_m_dev ratios {np.sum(y_m_dev==0), np.sum(y_m_dev==1)}")


    U_experiment = load_eigenvectors(eigenvector_path, debiasing_method)
#     U_experiment, s, vh = np.linalg.svd(A, full_matrices=True)

    results = defaultdict(dict)
    
    random.seed(0)
    np.random.seed(0)


    # biased_classifier = LinearSVC(fit_intercept=True, class_weight='balanced', dual=False, C=0.1, max_iter=10000)
    biased_classifier = LogisticRegression(penalty = "l1", C = 0.1, random_state = 0, solver = "saga", max_iter = 1000)


    biased_classifier.fit(x_train, y_m_train)
    y_pred_before = biased_classifier.predict(x_dev)
    biased_score = biased_classifier.score(x_dev, y_m_dev)

    f1_macro_before = f1_score(y_m_dev, y_pred_before, labels=None, average='macro')
    f1_micro_before = f1_score(y_m_dev, y_pred_before, labels=None, average='micro')
    _, tpr_gap_before = get_TPR_emoji(y_m_dev, biased_classifier.predict(x_dev), y_p_dev, y_m_train)
    _, tpr_gap_before_unweighted = get_TPR_emoji_unweighted(y_m_dev, biased_classifier.predict(x_dev), y_p_dev, y_m_train)

    if debiasing_method == "SAL":
        u_r = U_experiment[:, eigenvectors_to_remove:]
        proj = u_r @ u_r.T
        P = proj
    elif debiasing_method == "INLP":
        P = U_experiment

    debiased_x_train = P.dot(x_train.T).T
    debiased_x_dev = P.dot(x_dev.T).T
    
    random.seed(0)
    np.random.seed(0)


    # classifier = LinearSVC(fit_intercept=True, class_weight='balanced', dual=False, C=0.1, max_iter=10000)
    classifier = LogisticRegression(penalty = "l1", C = 0.1, random_state = 0, solver = "saga", max_iter = 1000)


    classifier.fit(debiased_x_train, y_m_train)

    y_pred_after = classifier.predict(debiased_x_dev)
    debiased_score = classifier.score(debiased_x_dev, y_m_dev)
    f1_macro_after = f1_score(y_m_dev, y_pred_after, labels=None, average='macro')
    f1_micro_after = f1_score(y_m_dev, y_pred_after, labels=None, average='micro')
    # debiased_score = biased_classifier.score(debiased_x_dev, y_m_dev)

    _, tpr_gap_after = get_TPR_emoji(y_m_dev, classifier.predict(debiased_x_dev), y_p_dev, y_m_train)
    _, tpr_gap_after_unweighted = get_TPR_emoji_unweighted(y_m_dev, classifier.predict(debiased_x_dev), y_p_dev, y_m_train)
    # results['biased_diff_tpr'] = tpr_gap_before 
    # results['debiased_diff_tpr'] = tpr_gap_after
    # results['biased_acc'] = biased_score
    # results['debiased_acc'] = debiased_score


    return f1_macro_before, f1_micro_before, f1_macro_after, f1_micro_after, biased_score, debiased_score, tpr_gap_before, tpr_gap_after, tpr_gap_before_unweighted, tpr_gap_after_unweighted 



#######################################################################
#
############### Functions used in biography experiments ###############
#
#######################################################################


def debiase_X(eigenvectors_to_remove, U, x_train, x_dev, x_test, debiasing_method):
    if debiasing_method == "SAL":
        u_r = U[:, eigenvectors_to_remove:]
        proj = u_r @ u_r.T
        P = proj
    elif debiasing_method == "INLP":
        P = U

    debiased_x_train = P.dot(x_train.T).T
    debiased_x_dev = P.dot(x_dev.T).T
    debiased_x_test = P.dot(x_test.T).T

    return debiased_x_train, debiased_x_dev, debiased_x_test



def tpr_exp(x_test, debiased_x_train, debiased_x_test, y_m_train, y_m_test, y_p_test, y_pred_before, p2i, i2p):

    # tprs_before, tprs_diff_before, mean_ratio_before = get_TPR(y_pred_before, y_m_test, p2i, i2p, y_p_test)
    # similarity_vs_tpr(tprs_diff_before, None, "before", "TPR", prof2fem)

    random.seed(0)
    np.random.seed(0)
    clf_debiased = LogisticRegression(warm_start = True, penalty = 'l2',
                             solver = "sag", multi_class = 'multinomial', fit_intercept = True,
                             verbose = 10, n_jobs = 90, random_state = 0, max_iter = 10)

    clf_debiased.fit(debiased_x_train, y_m_train)
    # debiased_accuracy_train = clf_debiased.score(debiased_x_train, y_m_train)
    debiased_accuracy_test = clf_debiased.score(debiased_x_test, y_m_test)
    # print(f"\n\n\n================================================\
    #             \nSCORE: of profession classifier on debiased train dataset\
    #             \n{debiased_accuracy_train}\
    #             \nSCORE: of profession classifier on debiased test dataset\
    #             \n{debiased_accuracy_test}\
    #             \n================================================\n\n\n")
    
    # y_pred_after_train = clf_debiased.predict(debiased_x_train)
    # tprs, tprs_diff_after_train, mean_ratio_after = get_TPR(y_pred_after_train, y_m_test, p2i, i2p, y_p_test)
    # similarity_vs_tpr(tprs_diff_before, None, "after", "TPR", prof2fem)
    
    y_pred_after_test = clf_debiased.predict(debiased_x_test)
    tprs, tprs_diff_after_test, mean_ratio_after, tpr_gap_after_test = get_TPR(y_pred_after_test, y_m_test, p2i, i2p, y_p_test)
    # similarity_vs_tpr(tprs_diff_after_test, None, "after", "TPR", prof2fem)

    # change_vals_after_train = np.array(list(tprs_diff_after_train.values()))
    # tpr_gap_after_train = rms(change_vals_after_train)
    # print("rms-diff after for train set: {}; rms-diff after for test set: {}".format(tpr_gap_after_train, tpr_gap_after_test))

    return debiased_accuracy_test, tpr_gap_after_test


def run_biography_experiments(dataset_path, eigenvector_path, eigenvectors_to_remove, debiasing_method):


    print("=========================================== Start of Biased Evaluation ===========================================")

    x_train, y_m_train, y_p_train, _, _, _, x_test, y_p_test, y_m_test = load_dataset(dataset_path)
    print(f"\n\nx_train shape {x_train.shape}, y_m_train {y_m_train.shape}, y_p_train {y_p_train.shape}, x_test {x_test.shape}, y_p_test {y_p_test.shape}, y_m_test {y_m_test.shape}\n\n")
    U_experiment = load_eigenvectors(eigenvector_path, debiasing_method)

    random.seed(0)
    np.random.seed(0)
    clf_original = LogisticRegression(warm_start = True, penalty = 'l2',
                            solver = "sag", multi_class = 'multinomial', fit_intercept = True,
                            verbose = 10, n_jobs = 90, random_state = 0, max_iter = 10)

    clf_original.fit(x_train, y_m_train)

    biased_accuracy = clf_original.score(x_test, y_m_test)
    y_pred_before = clf_original.predict(x_test)

    f1_macro_before = f1_score(y_m_test, y_pred_before, labels=None, average='macro')
    f1_micro_before = f1_score(y_m_test, y_pred_before, labels=None, average='micro')

    # print(f"Score of profession classifier on original(biased) dataset \n{biased_accuracy}")

    tprs_before, tprs_diff_before, mean_ratio_before, tpr_gap_before = get_TPR(y_pred_before, y_m_test, p2i, i2p, y_p_test)
    _, _, _, tpr_gap_before_unweighted = get_TPR_unweighted(y_pred_before, y_m_test, p2i, i2p, y_p_test)

    # similarity_vs_tpr(tprs_diff_before, None, "before", "TPR", prof2fem) 

    print("=========================================== Start of Debiased Evaluation ===========================================")

    if debiasing_method == "SAL":
        u_r = U_experiment[:, eigenvectors_to_remove:]
        proj = u_r @ u_r.T
        P = proj
    elif debiasing_method == "INLP":
        P = U_experiment

    debiased_x_train = P.dot(x_train.T).T
    debiased_x_test = P.dot(x_test.T).T

    random.seed(0)
    np.random.seed(0)
    clf_debiased = LogisticRegression(warm_start = True, penalty = 'l2',
                             solver = "sag", multi_class = 'multinomial', fit_intercept = True,
                             verbose = 10, n_jobs = 90, random_state = 0, max_iter = 10)

    clf_debiased.fit(debiased_x_train, y_m_train)
    debiased_accuracy = clf_debiased.score(debiased_x_test, y_m_test)
    y_pred_after = clf_debiased.predict(debiased_x_test)

    f1_macro_after = f1_score(y_m_test, y_pred_after, labels=None, average='macro')
    f1_micro_after = f1_score(y_m_test, y_pred_after, labels=None, average='micro')

    tprs_after, tprs_diff_after, mean_ratio_after, tpr_gap_after = get_TPR(y_pred_after, y_m_test, p2i, i2p, y_p_test)
    _, _, _, tpr_gap_after_unweighted = get_TPR_unweighted(y_pred_after, y_m_test, p2i, i2p, y_p_test)

    return f1_macro_before, f1_micro_before, f1_macro_after, f1_micro_after, biased_accuracy, debiased_accuracy, tpr_gap_before, tpr_gap_after, tpr_gap_before_unweighted, tpr_gap_after_unweighted


def generate_experiment_id(
    name,
    debiasing=None,
    model=None,
    experiment=None,
    assignment=None,
    assignment_ratio=None,
    seed=None,
    supervision_ratio=None,
):
    experiment_id = f"{name}"

    # Build the experiment ID.
    if isinstance(experiment, str):
        experiment_id += f"_e-{experiment}"
    if isinstance(model, str):
        experiment_id += f"_m-{model}"
    if isinstance(assignment_ratio, str):
        experiment_id += f"_r-{assignment_ratio}"
    if isinstance(debiasing, str):
        experiment_id += f"_d-{debiasing}"
    if isinstance(assignment, str):
        experiment_id += f"_a-{assignment}"
    if isinstance(seed, str):
        if seed != "null":
            experiment_id += f"_s-{seed}"
    if isinstance(supervision_ratio, str):
        if supervision_ratio != "null":
            experiment_id += f"_u-{supervision_ratio}"
    

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
        assignment_ratio=args.assignment_ratio,
        seed=args.seed,
        supervision_ratio=args.supervision_ratio,
    )

    p2i, i2p = load_dictionary("data/assignment/old/profession2index.txt")
    g2i, i2g = load_dictionary("data/assignment/old/gender2index.txt")

    if args.experiment == "biography" or args.experiment == "biography-labelled-by-overlap" or args.experiment == "biography-different-partial-n":
        f1_macro_before, f1_micro_before, f1_macro_after, f1_micro_after, biased_accuracy, debiased_accuracy, \
        tpr_gap_before, tpr_gap_after, tpr_gap_before_unweighted, tpr_gap_after_unweighted = \
        run_biography_experiments(args.dataset_path, args.projection_matrix_file, args.eigenvectors_to_remove, args.debiasing)
    elif args.experiment == "deepmoji":
        f1_macro_before, f1_micro_before, f1_macro_after, f1_micro_after, biased_accuracy, debiased_accuracy, tpr_gap_before, tpr_gap_after, tpr_gap_before_unweighted, tpr_gap_after_unweighted = \
        run_deepmoji_experiments(args.dataset_path, args.projection_matrix_file, args.eigenvectors_to_remove, args.debiasing, args.train_ratio, args.test_ratio)

    if args.experiment == "biography-different-partial-n":
        y_p_gold, y_p_learnt = load_true_Z_and_assignment_Z(f"{args.projection_matrix_path}/SAL_{args.bias}_{args.assignment}_np-{args.supervision_ratio}.mat")
        assignment_accuracy = np.sum(np.all(y_p_gold == y_p_learnt, axis=1)/y_p_gold.shape[0])
    else:
        assignment_accuracy=0
    
    print(f"biased_accuracy is {biased_accuracy},\n\
    debiased_accuracy_by_models is {debiased_accuracy},\n\
    tpr_gap_before is {tpr_gap_before},\n\
    tpr_gap_after_by_models is {tpr_gap_after}")

    result = []
    result.append(
        {
            "experiment_id": experiment_id,
            "assignment_accuracy": assignment_accuracy,
            "biased_accuracy": biased_accuracy,
            "debiased_accuracy": debiased_accuracy,
            "tpr_gap_before": tpr_gap_before,
            "tpr_gap_after": tpr_gap_after,
            "tpr_gap_before_unweighted": tpr_gap_before_unweighted,
            "tpr_gap_after_unweighted": tpr_gap_after_unweighted,
            "f1_macro_before": f1_macro_before,
            "f1_micro_before": f1_micro_before, 
            "f1_macro_after": f1_macro_after, 
            "f1_micro_after": f1_micro_after,
        }
    )

    # result = []
    # result.append({"biased_accuracy": biased_accuracy, "debiased_accuracy_by_models": debiased_accuracy_by_models,})

    os.makedirs(f"{args.save_path}", exist_ok=True)
    with open(f"{args.save_path}/{experiment_id}.json", "w") as f:
        json.dump(result, f)


