import glob
import json
import os
import pandas as pd
import numpy as np
import argparse


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
    choices=["Bert", "Fasttext", "Deepmoji05", "Deepmoji08", "Deepmoji80", "BertModel"],
    help="Models used to encode the context, e.g. BertModel",
)

parser.add_argument(
    "--tpr_gap_results_dir",
    action="store",
    type=str,
    help="Directory where all tpr gap results is stored.",
)


parser.add_argument(
    "--experiment",
    action="store",
    type=str,
    help="Experiment",
)


# parser.add_argument(
#     "--debiasing",
#     action="store",
#     type=str,
#     choices=["SAL", "INLP"],
#     help="The debiasing method",
# )

# parser.add_argument(
#     "--experiment",
#     action="store",
#     type=str,
#     choices=["ProfessionGender", "SentimentRace"],
#     help="The experiment name",
# )

# parser.add_argument(
#     "--assignment",
#     action="store",
#     type=str,
#     choices=["Kmeans", "Oracle", "Sal", "partialSup"],
#     help="The assignment used",
# )

def _parse_experiment_id(experiment_id):
    debiasing = None
    model = None
    experiment = None
    assignment = None
    bias = None
    supervision_ratio = None

    items = experiment_id.split("_")[1:]
    
    for item in items:
        id_, val = item[:1], item[2:]
        if id_ == "d":
            debiasing = val
        elif id_ == "m":
            model = val
        elif id_ == "e":
            experiment = val
        elif id_ == "a":
            assignment = val
        elif id_ == "b":
            bias = val
        elif id_ == "s":
            supervision_ratio = val
        else:
            raise ValueError(f"Unrecognized ID {id_}.")

    return debiasing, model, experiment, assignment, bias, supervision_ratio

def _pretty_model_name(row):
    pretty_name_mapping = {
        "SAL": r" + \textsc{SAL}",
        "INLP": r" + \textsc{INLP}",
    }

    pretty_assignment_mapping = {
        "Sal": r"\, + \textsc{AM}",
        "Kmeans": r"\, + \textsc{Kmeans}",
        "Oracle": r"\, + \textsc{Oracle}",
        "partialSup": r"\, + \textsc{Partial}",
    }

    pretty_name = pretty_assignment_mapping[row["assignment"]] + pretty_name_mapping[row["debiasing"]]
    
    if r"\textsc{AM}" in pretty_name:
        pretty_name = "\, + \textsc{AM" + f"{row['debiasing']}" + r"}"

    return pretty_name


# def _pretty_accuracy_value(row):
    
#     if row['biased_accuracy'] == row['debiased_accuracy']:
#         # return r"\ua{" + f"{row['debiased_accuracy']:.2f}" + r"}"
#         return f"{row['debiased_accuracy']:.2f}"
#     elif row['debiased_accuracy'] > row['biased_accuracy']:
#         return (
#             r"\uag{"
#             + f"{row['debiased_accuracy'] - row['biased_accuracy']:.2f}"
#             + r"} "
#             + f"{row['debiased_accuracy']:.2f}"
#         )
#     else:
#         return (
#             r"\dab{"
#             + f"{row['biased_accuracy'] - row['debiased_accuracy']:.2f}"
#             + r"} "
#             + f"{row['debiased_accuracy']:.2f}"
#         )

def _pretty_accuracy_value(row):
    
    if row['biased_accuracy'] == row['debiased_accuracy']:
        # return r"\ua{" + f"{row['debiased_accuracy']:.2f}" + r"}"
        return f"{row['debiased_accuracy']:.4f}"
    elif row['debiased_accuracy'] > row['biased_accuracy']:
        return (
            r"\uag{"
            + f"{row['debiased_accuracy'] - row['biased_accuracy']:.4f}"
            + r"} "
            + f"{row['debiased_accuracy']:.4f}"
        )
    else:
        return (
            r"\dab{"
            + f"{row['biased_accuracy'] - row['debiased_accuracy']:.4f}"
            + r"} "
            + f"{row['debiased_accuracy']:.4f}"
        )


def _pretty_assignment_acc_value(row):
    
    return f"{row['assignment_accuracy']:.6f}"


def _pretty_tpr_value(row, index_of_tpr):
    
    if row['tpr_gap_before'][index_of_tpr] == row['tpr_gap_after'][index_of_tpr]:
        # return r"\dab{" + f"{row['tpr_gap_after']:.2f}" + r"}"
        return f"{row['tpr_gap_after'][index_of_tpr]:.6f}"
    elif row['tpr_gap_after'][index_of_tpr] > row['tpr_gap_before'][index_of_tpr]:
        return (
            r"\ua{"
            + f"{row['tpr_gap_after'][index_of_tpr] - row['tpr_gap_before'][index_of_tpr]:.6f}"
            + r"} "
            + f"{row['tpr_gap_after'][index_of_tpr]:.6f}"
        )
    else:
        return (
            r"\da{"
            + f"{row['tpr_gap_before'][index_of_tpr] - row['tpr_gap_after'][index_of_tpr]:.6f}"
            + r"} "
            + f"{row['tpr_gap_after'][index_of_tpr]:.6f}"
        )


def _tpr_value(row, index_of_tpr):
    return f"{row['tpr_gap_after'][index_of_tpr]:.6f}"

def _pretty_f1_value(row, col_name):
    return f"{row[col_name]:.6f}"

if __name__ == "__main__":
    args = parser.parse_args()
        
    results = []
    for file_path in glob.glob(f"{args.persistent_dir}/src/assignment/results/{args.tpr_gap_results_dir}/*.json"):

        with open(file_path, "r") as f:
            results.extend(json.load(f))

    records = []
    for experiment in results:
        experiment_id = experiment["experiment_id"]
        # supervision_ratio = experiment["supervision_ratio"]
        assignment_accuracy = experiment["assignment_accuracy"]
        biased_accuracy = experiment["biased_accuracy"]
        debiased_accuracy = experiment["debiased_accuracy"]
        tpr_gap_before = experiment["tpr_gap_before"]
        tpr_gap_after = experiment["tpr_gap_after"]
        f1_macro_before = experiment['f1_macro_before']
        f1_micro_before = experiment['f1_micro_before']
        f1_macro_after = experiment['f1_macro_after']
        f1_micro_after = experiment['f1_micro_after']
        
        debiasing, model, experiment, assignment, bias, supervision_ratio = _parse_experiment_id(experiment["experiment_id"])
        
        records.append(
            {
                "experiment_id": experiment_id,
                "debiasing": debiasing,
                "supervision_ratio": supervision_ratio,
                "assignment_accuracy": assignment_accuracy,
                "model": model,
                "experiment": experiment,
                "assignment": assignment,
                "biased_accuracy": biased_accuracy,
                "debiased_accuracy": debiased_accuracy,
                "tpr_gap_before": tpr_gap_before,
                "tpr_gap_after": tpr_gap_after,
                "f1_macro_before": f1_macro_before, 
                "f1_micro_before": f1_micro_before, 
                "f1_macro_after": f1_macro_after, 
                "f1_micro_after": f1_micro_after,
            }
        )
        
    df = pd.DataFrame.from_records(records)

    # Filter to subset of results.
    df = df[df["model"] == args.model]

    # Get pretty model name.
    df["pretty_model_name"] = df.apply(lambda row: _pretty_model_name(row), axis=1)

    # Get pretty metric values.
    df["pretty_acc_value"] = df.apply(lambda row: _pretty_accuracy_value(row), axis=1)

    # Get pretty metric values.
    df["pretty_assignment_acc_value"] = df.apply(lambda row: _pretty_assignment_acc_value(row), axis=1)


    df['tpr-gap_value'] = df.apply(lambda row: _tpr_value(row, 0), axis=1)
    df['tpr-var_value'] = df.apply(lambda row: _tpr_value(row, 1), axis=1)

    # Get pretty metric values.
    df["pretty_tpr-var_value"] = df.apply(lambda row: _pretty_tpr_value(row, 0), axis=1)

    # Get pretty metric values.
    df["pretty_tpr-gap_value"] = df.apply(lambda row: _pretty_tpr_value(row, 1), axis=1)

    df["f1_macro_before"] = df.apply(lambda row: _pretty_f1_value(row, "f1_macro_before"), axis=1)
    df["f1_micro_before"] = df.apply(lambda row: _pretty_f1_value(row, "f1_micro_before"), axis=1)
    df["f1_macro_after"] = df.apply(lambda row: _pretty_f1_value(row, "f1_macro_after"), axis=1)
    df["f1_micro_after"] = df.apply(lambda row: _pretty_f1_value(row, "f1_micro_after"), axis=1)

    # To get proper ordering.
    df = df.sort_values(by="supervision_ratio")
    
    # To get proper ordering.
    df = df.sort_values(by="assignment")

    # To get proper ordering.
    df = df.sort_values(by="debiasing")


    df.reset_index(drop = True, inplace = True)
    df = df.reindex([0, 1, 2, 3, 4, 7, 6, 10, 8, 9, 5, 20, 19, 17, 15, 14, 12, 11, 13, 16, 18, 21])

    #define biased model frame
    df_original_model = pd.DataFrame({'pretty_model_name': r"\textsc{" + f"{args.model}" + r"}",
                        'pretty_acc_value': f"{df.iloc[0]['biased_accuracy']:.4f}",
                        'pretty_assignment_acc_value': f"100.0",
                        'pretty_tpr-var_value': f"{df.iloc[0]['tpr_gap_before'][0]:.6f}",
                        'pretty_tpr-gap_value': f"{df.iloc[0]['tpr_gap_before'][1]:.6f}",
                        "tpr-var_value": f"{df.iloc[0]['tpr_gap_before'][0]:.6f}",
                        "tpr-gap_value": f"{df.iloc[0]['tpr_gap_before'][1]:.6f}",
                        "debiased_accuracy": f"{df.iloc[0]['biased_accuracy']:.4f}",
                        "f1_macro_before": df.iloc[0]['f1_macro_before'],
                        "f1_micro_before":  df.iloc[0]['f1_micro_before']},
                        index=[0])

    df = pd.concat([df_original_model, df])

    with pd.option_context("max_colwidth", 1000):
        print(
            df.to_latex(
                float_format="%.3f",
                columns=[
                    "pretty_model_name",
                    "supervision_ratio",
                    "pretty_acc_value",
                    "pretty_assignment_acc_value",
                    "pretty_tpr-var_value",
                    "pretty_tpr-gap_value",
                    "f1_macro_before",
                    "f1_micro_before", 
                    "f1_macro_after", 
                    "f1_micro_after"
                ],
                index=False,
                escape=False,
            )
        )

    os.makedirs(f"{args.persistent_dir}/src/assignment/tables", exist_ok=True)

    with pd.option_context("max_colwidth", 1000):
        with open(
            f"{args.persistent_dir}/src/assignment/tables/tpr-table_m-{args.experiment}.tex",
            "w",
        ) as f:
            f.write(
                df.to_latex(
                    float_format="%.3f",
                    columns=[
                        "pretty_model_name",
                        "supervision_ratio",
                        "pretty_acc_value",
                        "pretty_assignment_acc_value",
                        "pretty_tpr-var_value",
                        "pretty_tpr-gap_value",
                        "f1_macro_before",
                        "f1_micro_before", 
                        "f1_macro_after", 
                        "f1_micro_after"
                    ],
                    index=False,
                    escape=False,
                )
            )
    



    with pd.ExcelWriter(f"{args.persistent_dir}/src/assignment/tables/tpr-table_m-{args.experiment}.xlsx") as writer:  
        df.to_excel(
            writer, 
            columns=[
                "pretty_model_name",
                "supervision_ratio",
                "debiased_accuracy",
                "pretty_assignment_acc_value",
                "tpr-var_value",
                "tpr-gap_value",
                "f1_macro_before",
                "f1_micro_before", 
                "f1_macro_after", 
                "f1_micro_after"
            ],
            sheet_name='twitter-different-partial-n')  
