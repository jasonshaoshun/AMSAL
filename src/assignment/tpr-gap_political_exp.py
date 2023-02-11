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
    choices=["BertModel", "Deepmoji"],
    help="Models used to encode the context, e.g. BertModel",
)

parser.add_argument(
    "--bias",
    action="store",
    type=str,
    choices=["age-gender", "age", "gender"],
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
    choices=["concat_twitter_sentiment", "concat_twitter_vulgar", "normal_twitter_sentiment", "normal_twitter_vulgar", "twitter-different-partial-n"],
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
    "--supervision_ratio",
    action="store",
    type=str,
    choices=["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"],
    help="ratio of the supervision",
)

parser.add_argument(
    "--dataset_path",
    action="store",
    type=str,
    help="The path to the dataset stored in npz",
)

parser.add_argument(
    "--eigenvector_path",
    action="store",
    type=str,
    help="The path to the projection matrix of INLP and SAL stored in npz/mat",
)

parser.add_argument(
    "--eigenvectors_to_keep",
    action="store",
    type=int,
    help="number of singular vectors to remove by SAL",
)

parser.add_argument(
    "--file_contains_original_Z_and_assignment_Z",
    action="store",
    type=str,
    help="file contains original_Z and assignment_Z",
)

    
def map_one_hot_2d_to_1d(y):

    y = map(lambda num: 0 if (num==[1, 0]).all() else 1, y)
    y = np.asarray(list(y))

    return y

def map_one_hot_3d_to_1d(y):

    y = map(lambda num: 0 if (num==[1, 0, 0]).all() else (1 if (num==[0, 1, 0]).all() else 2), y)
    y = np.asarray(list(y))

    return y

# def load_dictionary(path):
    
#     with open(path, "r", encoding = "utf-8") as f:
        
#         lines = f.readlines()
        
#     k2v, v2k = {}, {}
#     for line in lines:
        
#         k,v = line.strip().split("\t")
#         v = int(v)
#         k2v[k] = v
#         v2k[v] = k
    
#     return k2v, v2k



# def get_TPR(y_pred, y_true, p2i, i2p, y_p_test):
#     gender = np.where(y_p_test == 1, 'f', 'm')
#     scores = defaultdict(Counter)
#     prof_count_total = defaultdict(Counter)
    
#     for y_hat, y, g in zip(y_pred, y_true, gender):
        
#         if y == y_hat:
            
#             scores[i2p[y]][g] += 1
        
#         prof_count_total[i2p[y]][g] += 1
    
#     tprs = defaultdict(dict)
#     tprs_change = dict()
#     tprs_ratio = []
    
#     for profession, scores_dict in scores.items():
        
#         good_m, good_f = scores_dict["m"], scores_dict["f"]
#         prof_total_f = prof_count_total[profession]["f"]
#         prof_total_m = prof_count_total[profession]["m"]
#         if prof_total_m > 0 and prof_total_f > 0:
#             tpr_m = (good_m) / prof_total_m
#             tpr_f = (good_f) / prof_total_f

#             tprs[profession]["m"] = tpr_m
#             tprs[profession]["f"] = tpr_f
#             tprs_ratio.append(0)
#             tprs_change[profession] = tpr_f - tpr_m
#         else:
#             print(f"profession {profession} missed in tpr-calc:\n\
#                   number of {profession} male in test set {prof_total_m}\n\
#                   number of {profession} female in test set {prof_total_f}\n\n")
        
#     return tprs, tprs_change, np.mean(np.abs(tprs_ratio))
    
# def similarity_vs_tpr(tprs, word2vec, title, measure, prof2fem):
    
#     professions = list(tprs.keys())
#     #
#     """ 
#     sims = dict()
#     gender_direction = word2vec["he"] - word2vec["she"]
    
#     for p in professions:
#         sim = word2vec.cosine_similarities(word2vec[p], [gender_direction])[0]
#         sims[p] = sim
#     """
#     tpr_lst = [tprs[p] for p in professions]
#     sim_lst = [prof2fem[p] for p in professions]

#     #professions = [p.replace("_", " ") for p in professions if p in word2vec]
    
#     plt.plot(sim_lst, tpr_lst, marker = "o", linestyle = "none")
#     plt.xlabel("% women", fontsize = 13)
#     plt.ylabel(r'$GAP_{female,y}^{TPR}$', fontsize = 13)
#     for p in professions:
#         x,y = prof2fem[p], tprs[p]
#         plt.annotate(p , (x,y), size = 7, color = "red")
#     plt.ylim(-0.4, 0.55)
#     z = np.polyfit(sim_lst, tpr_lst, 1)
#     p = np.poly1d(z)
#     plt.plot(sim_lst,p(sim_lst),"r--")
#     plt.savefig("{}_vs_bias_{}_bert".format(measure, title), dpi = 600)
#     print("Correlation: {}; p-value: {}".format(*pearsonr(sim_lst, tpr_lst)))
#     plt.show()



# def get_TPR_emoji(y_main, y_hat_main, y_protected):
    
#     all_y = list(Counter(y_main).keys())
    
#     protected_vals = defaultdict(dict)
#     for label in all_y:
#         for i in range(2):
#             used_vals = (y_main == label) & (y_protected == i)
#             y_label = y_main[used_vals]
#             y_hat_label = y_hat_main[used_vals]
#             protected_vals['y:{}'.format(label)]['p:{}'.format(i)] = (y_label == y_hat_label).mean()
            
#     diffs = {}
#     for k, v in protected_vals.items():
#         vals = list(v.values())
#         diffs[k] = vals[0] - vals[1]
#     return protected_vals, diffs

# # def rms_diff(tpr_diff):
# #     return np.sqrt(np.mean(tpr_diff**2))

# def rms(arr):
#     return np.sqrt(np.mean(np.square(arr)))


def load_dataset(data_path, testset=True):

    saved_dataset = np.load(data_path)
    n_x = saved_dataset['x_train'].shape[0]
    n_train = np.floor(n_x * 0.7).astype(int)

    # x_train = saved_dataset['origin_tweet'][:n_train, :]
    x_train = saved_dataset['x_train'][:n_train, :]
    y_m_train = saved_dataset['y_m_train'][:n_train, :]
    y_p_train = saved_dataset['y_p_train'][:n_train, :]

    # x_test = saved_dataset['origin_tweet'][n_train:, :]
    x_test = saved_dataset['x_train'][n_train:, :]
    y_m_test = saved_dataset['y_m_train'][n_train:, :]
    y_p_test = saved_dataset['y_p_train'][n_train:, :]

    return x_train, y_m_train, y_p_train, x_test, y_m_test, y_p_test

def load_eigenvectors(data_path, debiasing_method):
    if debiasing_method == "SAL":
        saved_model = scipy.io.loadmat(f"{data_path}.mat")
        U_experiment = saved_model['U_experiment']
        # U_experiment = saved_model['U_best']
    elif debiasing_method == "INLP":
        saved_model = np.load(f"{data_path}.npz")
        U_experiment = saved_model['P']

    return U_experiment


def load_true_Z_and_assignment_Z(data_path):
    
    
    saved_model = scipy.io.loadmat(data_path)
    
    y_p_gold = saved_model['best_iter_original_Z']
    y_p_learnt = saved_model['best_iter_Z']
    
    return y_p_gold, y_p_learnt



def get_TPR(y_pred, y_true, protected_attribute, tpr_metric):

    if protected_attribute.shape[1] == 2:
        protected_attribute = map_one_hot_2d_to_1d(protected_attribute)
    elif protected_attribute.shape[1] == 3:
        protected_attribute = map_one_hot_3d_to_1d(protected_attribute)
    
    protected_attribute_dim = np.unique(protected_attribute).shape[0]
    
    scores = defaultdict(Counter)
    sent_count_total = defaultdict(Counter)
    
    # print(f"before y_pred is {y_pred}, y_true is {y_true}")
    y_pred = y_pred.reshape((-1, ))
    y_true = y_true.reshape((-1, ))
    print(f"y_pred is {y_pred},\ny_true is {y_true}")
    protected_attribute = protected_attribute.reshape((-1, ))
    for y_hat, y, g in zip(y_pred, y_true, protected_attribute):
        if y == y_hat:
            scores[y][g] += 1
        sent_count_total[y][g] += 1
    print(f"\n\nscores is {scores}, sent_count_total is {sent_count_total}\n\n")
    tprs = defaultdict(dict)
    tprs_change = dict()

    for sentiment, scores_dict in scores.items():

        if len(sent_count_total[sentiment]) == protected_attribute_dim:
            for protect_attribute_class in sent_count_total[sentiment]:
                tprs[sentiment][protect_attribute_class] = scores_dict[protect_attribute_class] / sent_count_total[sentiment][protect_attribute_class]
                print(f"\n\nsentiment is {sentiment}, protect_attribute_class is {protect_attribute_class}, tprs[sentiment] is {tprs[sentiment]}, \
                scores_dict[protect_attribute_class] is {scores_dict[protect_attribute_class]}, \
                sent_count_total[sentiment][protect_attribute_class] is {sent_count_total[sentiment][protect_attribute_class]}\n\n")
                
            if tpr_metric == 'gap':
                tprs_on_same_sentiment = np.asarray(list(tprs[sentiment].values()))
                print(f"tprs_on_same_sentiment is {tprs_on_same_sentiment}")
                tprs_change[sentiment] = (tprs_on_same_sentiment[0] - tprs_on_same_sentiment[1])
            elif tpr_metric == 'variance':
                tprs_change[sentiment] = np.var(np.asarray(list(tprs[sentiment].values())))
            else:
                sys.exit("tpr_metric is not found")
        else:    
            print(f"sent_count_total[sentiment] {sent_count_total[sentiment]}\n\
            sentiment {sentiment} missed in tpr calculation\n")

    print(f"tprs_change is {tprs_change}")

    if tpr_metric == 'gap':
        tprs_change = np.sqrt(np.mean(np.square(np.asarray(list(tprs_change.values())))))
    elif tpr_metric == 'variance':
        tprs_change = np.sum(np.array(list(tprs_change.values())))

    return tprs, tprs_change



# def run_deepmoji_experiments(dataset_path, eigenvector_path, eigenvectors_to_keep, debiasing_method):

#     x_train, y_m_train, y_p_train, x_dev, y_p_dev, y_m_dev = load_dataset(dataset_path, testset = False)

#     U_experiment = load_eigenvectors(eigenvector_path, debiasing_method)

#     results = defaultdict(dict)

#     biased_classifier = LinearSVC(fit_intercept=True, class_weight='balanced', dual=False, C=0.1, max_iter=10000)
#     biased_classifier.fit(x_train, y_m_train)
#     biased_score = biased_classifier.score(x_dev, y_m_dev)
#     _, biased_diffs = get_TPR_emoji(y_m_dev, biased_classifier.predict(x_dev), y_p_dev)

#     if debiasing_method == "SAL":
#         u_r = U_experiment[:, eigenvectors_to_keep:]
#         proj = u_r @ u_r.T
#         P = proj
#     elif debiasing_method == "INLP":
#         P = U_experiment

#     debiased_x_train = P.dot(x_train.T).T
#     debiased_x_dev = P.dot(x_dev.T).T

#     classifier = LinearSVC(fit_intercept=True, class_weight='balanced', dual=False, C=0.1, max_iter=10000)

#     classifier.fit(debiased_x_train, y_m_train)
#     debiased_score = classifier.score(debiased_x_dev, y_m_dev)

#     p_classifier = SGDClassifier(warm_start=True, loss='log', n_jobs=64, max_iter=10000, random_state=0, tol=1e-3)
#     p_classifier.fit(debiased_x_train, y_p_train)
#     p_score = p_classifier.score(debiased_x_dev, y_p_dev)

#     _, debiased_diffs = get_TPR_emoji(y_m_dev, classifier.predict(debiased_x_dev), y_p_dev)


#     results['p_acc'] = p_score
#     #     results[ratio]['biased_diff_tpr'] = biased_diffs['y:0']
#     results['biased_diff_tpr'] = rms(list(biased_diffs.values()))
#     #     results[ratio]['debiased_diff_tpr'] = debiased_diffs['y:0']
#     results['debiased_diff_tpr'] = rms(list(debiased_diffs.values()))

#     results['biased_acc'] = biased_score
#     results['debiased_acc'] = debiased_score
        
#     return results['biased_acc'], results['debiased_acc'], results['biased_diff_tpr'], results['debiased_diff_tpr']



# #######################################################################
# #
# ############### Functions used in biography experiments ###############
# #
# #######################################################################


def debiase_X(eigenvectors_to_keep, U, x_train, x_test, debiasing_method):
    if debiasing_method == "SAL":
        u_r = U[:, eigenvectors_to_keep:]
        proj = u_r @ u_r.T
        P = proj
    elif debiasing_method == "INLP":
        P = U

    debiased_x_train = P.dot(x_train.T).T
    debiased_x_test = P.dot(x_test.T).T
    return debiased_x_train, debiased_x_test


# def tpr_exp(x_test, debiased_x_train, debiased_x_test, y_m_train, y_m_test, y_p_test, y_pred_before, p2i, i2p):

#     tprs_before, tprs_diff_before, mean_ratio_before = get_TPR(y_pred_before, y_m_test, p2i, i2p, y_p_test)
#     # similarity_vs_tpr(tprs_diff_before, None, "before", "TPR", prof2fem)

#     random.seed(0)
#     np.random.seed(0)
#     clf_debiased = LogisticRegression(warm_start = True, penalty = 'l2',
#                              solver = "sag", multi_class = 'multinomial', fit_intercept = True,
#                              verbose = 10, n_jobs = 90, random_state = 0, max_iter = 10)

#     clf_debiased.fit(debiased_x_train, y_m_train)
#     debiased_accuracy_train = clf_debiased.score(debiased_x_train, y_m_train)
#     debiased_accuracy_test = clf_debiased.score(debiased_x_test, y_m_test)
#     # print(f"\n\n\n================================================\
#     #             \nSCORE: of profession classifier on debiased train dataset\
#     #             \n{debiased_accuracy_train}\
#     #             \nSCORE: of profession classifier on debiased test dataset\
#     #             \n{debiased_accuracy_test}\
#     #             \n================================================\n\n\n")
    
#     y_pred_after_train = clf_debiased.predict(debiased_x_train)
#     tprs, tprs_diff_after_train, mean_ratio_after = get_TPR(y_pred_after_train, y_m_test, p2i, i2p, y_p_test)
#     # similarity_vs_tpr(tprs_diff_before, None, "after", "TPR", prof2fem)
    
#     y_pred_after_test = clf_debiased.predict(debiased_x_test)
#     tprs, tprs_diff_after_test, mean_ratio_after = get_TPR(y_pred_after_test, y_m_test, p2i, i2p, y_p_test)
#     # similarity_vs_tpr(tprs_diff_after_test, None, "after", "TPR", prof2fem)

#     change_vals_after_train = np.array(list(tprs_diff_after_train.values()))
#     change_vals_after_test = np.array(list(tprs_diff_after_test.values()))
#     tpr_gap_after_train = rms(change_vals_after_train)
#     tpr_gap_after_test = rms(change_vals_after_test)
#     print("rms-diff after for train set: {}; rms-diff after for test set: {}".format(tpr_gap_after_train, tpr_gap_after_test))

#     return debiased_accuracy_test, tpr_gap_after_test


# def run_twitter_experiments(dataset_path, eigenvector_path, eigenvectors_to_keep, debiasing_method):

#     x_train, y_m_train, y_p_train, x_test, y_m_test, y_p_test = load_dataset(dataset_path)
#     U_experiment = load_eigenvectors(eigenvector_path, debiasing_method)


#     print("=========================================== Start of BIASED experiments ===========================================")

#     random.seed(0)
#     np.random.seed(0)
#     clf_original = LogisticRegression(warm_start = True, penalty = 'l2',
#                             solver = "sag", multi_class = 'multinomial', fit_intercept = True,
#                             verbose = 10, n_jobs = 90, random_state = 0, max_iter = 100)

#     clf_original.fit(x_train, y_m_train)
#     # print(f"y_m_train is {y_m_train}, y_m_test is {y_m_test}")
#     biased_accuracy = clf_original.score(x_test, y_m_test)
#     print(f"Score of profession classifier on original(biased) dataset \n{biased_accuracy}")
#     y_pred_before = clf_original.predict(x_test)

#     tprs_before, tprs_change_before = get_TPR(y_pred_before, y_m_test, y_p_test)
#     # print(f"tprs_change_before {np.array(list(tprs_change_before.values()))}")
#     # tpr_gap_before = np.sum(np.array(list(tprs_change_before.values())))
#     tpr_gap_before = tprs_change_before

#     print("=========================================== Start of DeBIASED experiments ===========================================")

#     debiased_x_train, debiased_x_test = debiase_X(eigenvectors_to_keep, U_experiment, x_train, x_test, debiasing_method)

#     random.seed(0)
#     np.random.seed(0)
#     clf_original = LogisticRegression(warm_start = True, penalty = 'l2',
#                             solver = "sag", multi_class = 'multinomial', fit_intercept = True,
#                             verbose = 10, n_jobs = 90, random_state = 0, max_iter = 100)

#     clf_original.fit(debiased_x_train, y_m_train)

#     debiased_accuracy = clf_original.score(debiased_x_test, y_m_test)
#     # print(f"Score of profession classifier on original(biased) dataset \n{debiased_accuracy}")
#     y_pred_after = clf_original.predict(debiased_x_test)

#     tprs_after, tprs_change_after = get_TPR(y_pred_after, y_m_test, y_p_test)

#     # tpr_gap_after = np.sum(np.array(list(tprs_change_after.values())))
#     tpr_gap_after = tprs_change_after

#     # print(f"tprs_change_after {np.array(list(tprs_change_after.values()))}, sum {tpr_gap_after}")

#     return biased_accuracy, debiased_accuracy, tpr_gap_before, tpr_gap_after


def map_one_hot_to_1d(y_m_train, y_m_test):

    # y_p_train = map(lambda num: 1 if (num==[1, 0, 0, 1, 0]).all() else (2 if (num==[1, 0, 0, 0, 1]).all() \
    # else (3 if (num==[0, 1, 0, 1, 0]).all() else(4 if (num==[0, 1, 0, 0, 1]).all() \
    # else (5 if (num==[0, 0, 1, 1, 0]).all() else 6)))), y_p_train)

    # y_p_test = map(lambda num: 1 if (num==[1, 0, 0, 1, 0]).all() else (2 if (num==[1, 0, 0, 0, 1]).all() \
    # else (3 if (num==[0, 1, 0, 1, 0]).all() else(4 if (num==[0, 1, 0, 0, 1]).all() \
    # else (5 if (num==[0, 0, 1, 1, 0]).all() else 6)))), y_p_test)

    y_m_train = map(lambda num: 0 if (num==[1, 0, 0]).all() else (1 if (num==[0, 1, 0]).all() else 2), y_m_train)
    y_m_train = np.asarray(list(y_m_train))
    y_m_test = map(lambda num: 0 if (num==[1, 0, 0]).all() else (1 if (num==[0, 1, 0]).all() else 2), y_m_test)
    y_m_test = np.asarray(list(y_m_test))

    return y_m_train, y_m_test




def run_political_twitter_experiments(dataset_path, eigenvector_path, eigenvectors_to_keep, debiasing_method):

    x_train, y_m_train, y_p_train, x_test, y_m_test, y_p_test = load_dataset(dataset_path)
    U_experiment = load_eigenvectors(eigenvector_path, debiasing_method)
    # print(f"before y_m_train {y_m_train}")
    # y_p_train, y_p_test = map_one_hot_to_1d(y_p_train, y_p_test)
    # print(f"after y_m_train {y_m_train}")

    print("=========================================== Start of BIASED experiments ===========================================")

    random.seed(0)
    np.random.seed(0)
    clf_original = LogisticRegression(warm_start = True, penalty = 'l2',
                            solver = "sag", multi_class = 'multinomial', fit_intercept = True,
                            verbose = 10, n_jobs = 90, random_state = 0, max_iter = 100)

    clf_original.fit(x_train, y_m_train)
    # print(f"y_m_train is {y_m_train}, y_m_test is {y_m_test}")
    biased_accuracy = clf_original.score(x_test, y_m_test)
    print(f"Score of profession classifier on original(biased) dataset \n{biased_accuracy}")
    y_pred_before = clf_original.predict(x_test)
    f1_macro_before = f1_score(y_m_test, y_pred_before, labels=None, average='macro')
    f1_micro_before = f1_score(y_m_test, y_pred_before, labels=None, average='micro')

    tprs_before_age, tprs_change_before_age = get_TPR(y_pred_before, y_m_test, y_p_test[:, 0:3], 'variance')
    tprs_before_gender, tprs_change_before_gender = get_TPR(y_pred_before, y_m_test, y_p_test[:, 3:5], 'gap')
    tpr_gap_before = [tprs_change_before_age, tprs_change_before_gender]

    print("=========================================== Start of DeBIASED experiments ===========================================")

    debiased_x_train, debiased_x_test = debiase_X(eigenvectors_to_keep, U_experiment, x_train, x_test, debiasing_method)

    random.seed(0)
    np.random.seed(0)
    clf_original = LogisticRegression(warm_start = True, penalty = 'l2',
                            solver = "sag", multi_class = 'multinomial', fit_intercept = True,
                            verbose = 10, n_jobs = 90, random_state = 0, max_iter = 100)

    clf_original.fit(debiased_x_train, y_m_train)

    debiased_accuracy = clf_original.score(debiased_x_test, y_m_test)
    # print(f"Score of profession classifier on original(biased) dataset \n{debiased_accuracy}")
    y_pred_after = clf_original.predict(debiased_x_test)

    tprs_after_age, tprs_change_after_age = get_TPR(y_pred_after, y_m_test, y_p_test[:, 0:3], 'variance')
    tprs_after_gender, tprs_change_after_gender = get_TPR(y_pred_after, y_m_test, y_p_test[:, 3:5], 'gap')

    # tpr_gap_after = np.sum(np.array(list(tprs_change_after.values())))
    tpr_gap_after = [tprs_change_after_age, tprs_change_after_gender]

    # print(f"tprs_change_after {np.array(list(tprs_change_after.values()))}, sum {tpr_gap_after}")

    f1_macro_after = f1_score(y_m_test, y_pred_after, labels=None, average='macro')
    f1_micro_after = f1_score(y_m_test, y_pred_after, labels=None, average='micro')

    return f1_macro_before, f1_micro_before, f1_macro_after, f1_micro_after, biased_accuracy, debiased_accuracy, tpr_gap_before, tpr_gap_after




def generate_experiment_id(
    name,
    debiasing=None,
    model=None,
    experiment=None,
    assignment=None,
    bias=None,
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
    if isinstance(supervision_ratio, str):
        experiment_id += f"_s-{supervision_ratio}"
    
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
        supervision_ratio=args.supervision_ratio,
    )

    # eigenvectors_to_keep = 2

    # dataset_to_dataset_folder = {
    #     "BertModel": f"/data/{args.experiment}/{args.model}/{args.model}_{args.bias}.npz",
    #     "Deepmoji": f"/data/{args.experiment}/{args.model}/{args.model}_{args.bias}.npz"
    # }

    # dataset_to_eigenvector_folder = {
    #     "BertModel": f"data/projection_matrix/{args.experiment}/{args.model}/{args.bias}_{args.assignment}.mat",
    #     "Deepmoji": f"data/projection_matrix/{args.experiment}/{args.model}/{args.bias}_{args.assignment}.mat"
    # }

    # if args.model == "BertModel":
    #     biased_accuracy, debiased_accuracy, tpr_gap_before, tpr_gap_after = run_twitter_experiments(\
    #     f"{args.persistent_dir}/{dataset_to_dataset_folder[args.model]}", f"{args.persistent_dir}/{dataset_to_eigenvector_folder[args.model]}",\
    #     eigenvectors_to_keep, args.debiasing)
    # else:
    #     print(f"The model is unknown, can't find appropriate test for the model and dataset")
    

    # if args.debiasing == "SAL":
    #     y_p_gold, y_p_learnt = load_true_Z_and_assignment_Z(f"{args.eigenvector_path}/../SAL_{args.bias}_{args.assignment}_np-{args.supervision_ratio}.mat")
    # elif args.debiasing == "INLP":
    #     y_p_gold, y_p_learnt = load_true_Z_and_assignment_Z(f"{args.eigenvector_path}/../SAL_{args.bias}_{args.assignment}_np-{args.supervision_ratio}.mat")
    y_p_gold, y_p_learnt = load_true_Z_and_assignment_Z(args.file_contains_original_Z_and_assignment_Z)


    assignment_accuracy = np.sum(np.all(y_p_gold == y_p_learnt, axis=1)/y_p_gold.shape[0])

    f1_macro_before, f1_micro_before, f1_macro_after, f1_micro_after, biased_accuracy, debiased_accuracy, tpr_gap_before, tpr_gap_after = run_political_twitter_experiments(\
    args.dataset_path, args.eigenvector_path, args.eigenvectors_to_keep, args.debiasing)

    print(f"assignment_accuracy is {assignment_accuracy},\n\
    biased_accuracy is {biased_accuracy},\n\
    debiased_accuracy is {debiased_accuracy},\n\
    tpr_gap_before is {tpr_gap_before},\n\
    tpr_gap_after is {tpr_gap_after}")

    result = []
    result.append(
        {
            "experiment_id": experiment_id,
            "assignment_accuracy": assignment_accuracy,
            "biased_accuracy": biased_accuracy,
            "debiased_accuracy": debiased_accuracy,
            "tpr_gap_before": tpr_gap_before,
            "tpr_gap_after": tpr_gap_after,
            "supervision_ratio": args.supervision_ratio,
            "f1_macro_before": f1_macro_before, 
            "f1_micro_before": f1_micro_before, 
            "f1_macro_after": f1_macro_after, 
            "f1_micro_after": f1_micro_after,
        }
    )

    # result = []
    # result.append({"biased_accuracy": biased_accuracy, "debiased_accuracy_by_models": debiased_accuracy_by_models,})

    # os.makedirs(f"{args.persistent_dir}/src/assignment/results/political_twitter/tpr-gap", exist_ok=True)
    # with open(f"{args.persistent_dir}/src/assignment/results/political_twitter/tpr-gap/{experiment_id}.json", "w") as f:
    #     json.dump(result, f)
        
    os.makedirs(f"{args.persistent_dir}/src/assignment/results/{args.experiment}/tpr-gap", exist_ok=True)
    with open(f"{args.persistent_dir}/src/assignment/results/{args.experiment}/tpr-gap/{experiment_id}.json", "w") as f:
        json.dump(result, f)
