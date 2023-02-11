import numpy as np
import scipy.io
import h5py
import torch

all_models=["BertModel", "AlbertModel", "RobertaModel", "GPT2Model"]
all_bias_types=["gender", "race", "religion"]

for model_name in all_models:
    for bias_type in all_bias_types:
        
        # ====================== new_sal ====================== 
        saved_model = scipy.io.loadmat('data/projection_matrix/{}/{}.mat'.format(model_name, bias_type))

        UU = np.dot(saved_model['U_experiment'][:, 2:], saved_model['U_experiment'][:, 2:].T)
        P = torch.tensor(UU, dtype=torch.float32)
        torch.save(P, "data/projection_matrix/{}/{}_experiment.pt".format(model_name, bias_type),)

        UU = np.dot(saved_model['U_best'][:, 2:], saved_model['U_best'][:, 2:].T)
        P = torch.tensor(UU, dtype=torch.float32)
        torch.save(P, "data/projection_matrix/{}/{}_best.pt".format(model_name, bias_type),)

        # ====================== Kmeans ====================== 
        saved_model = scipy.io.loadmat('data/projection_matrix/{}/{}_kmeans.mat'.format(model_name, bias_type))

        UU = np.dot(saved_model['U_experiment'][:, 2:], saved_model['U_experiment'][:, 2:].T)
        P = torch.tensor(UU, dtype=torch.float32)
        torch.save(P, "data/projection_matrix/{}/{}_kmeans.pt".format(model_name, bias_type),)
