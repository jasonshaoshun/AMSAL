import numpy as np
import scipy.io
import h5py
import torch

experiments = "political"
all_models=["BertModel", "AlbertModel", "RobertaModel", "GPT2Model"]
all_bias_types=["gender", "race", "religion"]

for model_name in all_models:
    for bias_type in all_bias_types:
        print(f"model_name {model_name}, bias_type {bias_type}")

        saved_model = np.load('data/{}/{}/{}.npz'.format(experiments, model_name, bias_type))
        x_dev = saved_model['x_dev']
        x_test = saved_model['x_test']
        y_p_dev = saved_model['y_p_dev'].astype(int)
        y_p_test = saved_model['y_p_test'].astype(int)

        # ====================== SAL ====================== 
        saved_model = scipy.io.loadmat('data/projection_matrix/{}/{}/{}_Sal.mat'.format(experiments, model_name, bias_type))
        x_train = saved_model['best_iter_X']
        y_p_train = saved_model['best_iter_Z'].astype(int)
        y_p_train = y_p_train[:, 0]
        y_p_train_gold = saved_model['best_iter_original_Z'].astype(int)
        y_p_train_gold = y_p_train_gold[:, 0]

        np.savez(f"data/projection_matrix/{experiments}/{model_name}/{bias_type}_Sal.npz", \
            x_train = x_train, x_dev = x_dev, x_test = x_test, \
            y_p_train = y_p_train, y_p_dev = y_p_dev, y_p_test = y_p_test, \
            y_p_train_gold = y_p_train_gold)


        # ====================== Oracle ====================== 
        saved_model = scipy.io.loadmat('data/projection_matrix/{}/{}/{}_Oracle.mat'.format(experiments, model_name, bias_type))
        x_train = saved_model['best_iter_X']
        y_p_train = saved_model['best_iter_Z'].astype(int)
        y_p_train = y_p_train[:, 0]

        np.savez(f"data/projection_matrix/{experiments}/{model_name}/{bias_type}_Oracle.npz", \
            x_train = x_train, x_dev = x_dev, x_test = x_test, \
            y_p_train = y_p_train, y_p_dev = y_p_dev, y_p_test = y_p_test, \
            y_p_train_gold = y_p_train)
        
        
        # ====================== partialSup ====================== 
        saved_model = scipy.io.loadmat('data/projection_matrix/{}/{}/{}_partialSup.mat'.format(experiments, model_name, bias_type))
        x_train = saved_model['best_iter_X']
        y_p_train = saved_model['best_iter_Z'].astype(int)
        y_p_train = y_p_train[:, 0]
        y_p_train_gold = saved_model['best_iter_original_Z'].astype(int)
        y_p_train_gold = y_p_train_gold[:, 0]

        np.savez(f"data/projection_matrix/{experiments}/{model_name}/{bias_type}_partialSup.npz", \
            x_train = x_train, x_dev = x_dev, x_test = x_test, \
            y_p_train = y_p_train, y_p_dev = y_p_dev, y_p_test = y_p_test, \
            y_p_train_gold = y_p_train_gold)


        # ====================== Kmeans ====================== 
        saved_model = scipy.io.loadmat('data/projection_matrix/{}/{}/{}_Kmeans.mat'.format(experiments, model_name, bias_type))
        x_train = saved_model['kmeans_best_iter_X']
        y_p_train = saved_model['kmeans_best_iter_Z'].astype(int)
        y_p_train = y_p_train[:, 0]
        y_p_train_gold = saved_model['kmeans_best_iter_original_Z'].astype(int)
        y_p_train_gold = y_p_train_gold[:, 0]

        np.savez(f"data/projection_matrix/{experiments}/{model_name}/{bias_type}_Kmeans.npz", \
            x_train = x_train, x_dev = x_dev, x_test = x_test, \
            y_p_train = y_p_train, y_p_dev = y_p_dev, y_p_test = y_p_test, \
            y_p_train_gold = y_p_train_gold)