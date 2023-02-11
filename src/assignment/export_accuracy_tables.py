import glob
import json
import os
import pandas as pd
import numpy as np
import argparse
import scipy.io


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
    choices=["Bert", "Fasttext", "Deepmoji"],
    help="Models used to encode the context, e.g. BertModel",
)

parser.add_argument(
    "--assignment_range",
    nargs='+',
    help="List of assignments that we would like to test the accuracy matched with tru label",
)

def load_assignment_return_accuracy(assignment_path):

    saved_dataset = scipy.io.loadmat(assignment_path)

    x_train = saved_dataset['best_iter_X']
    y_p_train_assignment = saved_dataset['best_iter_Z']
    y_p_train = saved_dataset['best_iter_original_Z']
    y_m_train = saved_dataset['best_iter_Y']

    if len(np.unique(y_m_train)) > len(np.unique(y_p_train_assignment)):
        alignment_Y_Z = 0
    else:
        alignment_Y_Z = np.count_nonzero(y_m_train[:, 0] == y_p_train_assignment[:, 0]) / y_m_train.shape[0]
        alignment_Y_Z = np.max([alignment_Y_Z, 1 - alignment_Y_Z])
    
    # print(f"alignment_Y_Z is {alignment_Y_Z}")
    # print(f"y_m_train shape {y_m_train.shape}")

    # print(f"y_p_train_assignment {y_p_train_assignment.shape}, y_p_train {y_p_train.shape}")
    # print(f"count {np.count_nonzero(y_p_train_assignment[:, 0] == y_p_train[:, 0])}, num_train {y_p_train_assignment.shape[0]}")
    acc = np.count_nonzero(y_p_train_assignment[:, 0] == y_p_train[:, 0]) / y_p_train_assignment.shape[0]

    return np.max([acc, 1 - acc]), alignment_Y_Z

if __name__ == "__main__":
    args = parser.parse_args()
    
    dataset_to_eigenvector_folder = {
        "Bert": "data/assignment/projection_matrix/BERT/",
        "Fasttext": "data/assignment/projection_matrix/FastText/",
        "Deepmoji": "data/assignment/projection_matrix/05_all/"
    }

    # df_original_model = pd.DataFrame({'pretty_model_name': r"\textsc{" + f"{args.model}" + r"}",
    #                     'accuracy': f"{acc:.2f}"},
    #                     index=[0])

    records = []
    for assignment in args.assignment_range:
        assignment_path = f"{args.persistent_dir}/{dataset_to_eigenvector_folder[args.model]}/SAL_{assignment}_assignment.mat"
        acc, alignment_Y_Z = load_assignment_return_accuracy(assignment_path)

        records.append(
            {
                "pretty_model_name": r"\textsc{" + f"{args.model}" + r"}",
                "assignment_method": r"\textsc{" + f"{assignment}" + r"}",
                "accuracy": f"{acc:.2f}",
                "alignment_Y_Z": f"{alignment_Y_Z:.2f}",
            }
        )
        
    df = pd.DataFrame.from_records(records)

    with pd.option_context("max_colwidth", 1000):
        print(
            df.to_latex(
                float_format="%.3f",
                columns=[
                    "pretty_model_name",
                    "assignment_method",
                    "accuracy",
                    "alignment_Y_Z"
                ],
                index=False,
                escape=False,
            )
        )

    os.makedirs(f"{args.persistent_dir}/src/assignment/tables", exist_ok=True)

    with pd.option_context("max_colwidth", 1000):
        with open(
            f"{args.persistent_dir}/src/assignment/tables/accuracy-table_m-{args.model}.tex",
            "w",
        ) as f:
            f.write(
                df.to_latex(
                    float_format="%.3f",
                    columns=[
                        "pretty_model_name",
                        "assignment_method",
                        "accuracy",
                        "alignment_Y_Z"
                    ],
                    index=False,
                    escape=False,
                )
            )
