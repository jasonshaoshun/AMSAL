import argparse
import numpy as np
import scipy.io
import h5py
import torch
import os
from bias_bench.util import generate_experiment_id

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Computes the projection matrix for INLP.")

parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)

# parser.add_argument(
#     "--experiments",
#     action="store",
#     type=str,
#     help="The experiments that working on, same name on the folder that contains the dataset to compute svds",
# )

parser.add_argument(
    "--pre_assignment_path",
    action="store",
    type=str,
    default="data/saved_dataset/BertModel/gender_assignment.npz",
    help="The path to dataset created by new sal assignment",
)

parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="SALBertModel",
    choices=["SALBertModel", "SALAlbertModel", "SALRobertaModel", "SALGPT2Model"],
    help="Model (e.g., BertModel) to compute the INLP projection matrix for. "
    "Typically, these correspond to a HuggingFace class.",
)

parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-uncased",
    choices=["bert-base-uncased", "albert-base-v2", "roberta-base", "gpt2"],
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.",
)

parser.add_argument(
    "--bias_type",
    action="store",
    default="gender",
    choices=["gender", "race", "religion"],
    help="What type of bias to compute the INLP projection matrix for.",
)

parser.add_argument(
    "--pre_assignment_type",
    action="store",
    type=str,
    default="Oracle",
    choices=["partialSup", "Sal", "Kmeans", "Oracle"],
    help="compute inlp for kmeans(True) or sal(False)",
)

parser.add_argument("--seed", action="store", type=int, default=0, help="Seed for RNG.")

if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name="projection",
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        bias_type=args.bias_type,
        seed=args.seed,
        pre_assignment_type=args.pre_assignment_type,
    )

    print("Computing projection matrix:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - bias_type: {args.bias_type}")
    print(f" - seed: {args.seed}")
    print(f" - pre_assignment_type: {args.pre_assignment_type}")

    model_to_folder = {
        "SALBertModel": "BertModel",
        "SALAlbertModel": "AlbertModel",
        "SALRobertaModel": "RobertaModel",
        "SALGPT2Model": "GPT2Model"
    }

    print(f"path is {args.pre_assignment_path}")
    saved_model = np.load(args.pre_assignment_path)
    x_train = saved_model['x_train']
    y_p_train = saved_model['y_p_train']
    y_p_train_2d = np.asarray([y_p_train, 1 - y_p_train]).T

    A = np.dot(x_train.T, y_p_train_2d) / x_train.shape[0]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    removal = 2
    u_r = u[:, removal:]
    proj = u_r @ u_r.T
    P = torch.tensor(proj, dtype=torch.float32)

    print(
        f"Saving computed projection matrix to: {args.persistent_dir}/results/eigenvectors/{experiment_id}.pt"
    )

    os.makedirs(f"{args.persistent_dir}/results/eigenvectors", exist_ok=True)

    torch.save(
        P,
        f"{args.persistent_dir}/results/eigenvectors/{experiment_id}.pt",
    )