import glob
import json
import os
import pandas as pd
import numpy as np
import argparse
import os.path
import openpyxl

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
    "--tpr_gap_results_dir",
    action="store",
    type=str,
    help="Directory where all tpr gap results is stored.",
)

# parser.add_argument(
#     "--supervision_ratio",
#     action="store",
#     type=str,
#     help="ratio of positive and negative samples in main label or protected attribute",
# )

parser.add_argument(
    "--save_path",
    action="store",
    type=str,
    help="the path to save the tex file and spreadsheet",
)

parser.add_argument(
    "--sheet_name",
    action="store",
    type=str,
    help="save the experiment result in spreadsheet, the sheetname of the result",
)

parser.add_argument(
    "--summary_xlsx_path",
    action="store",
    type=str,
    help="the path to to speardsheet",
)

# parser.add_argument(
#     "--num_tpr_category",
#     action="store",
#     type=int,
#     help="number of tpr categories, we use tpr-gap for deepmoji and biography experiments,\
#     but used tpr-gap and tpr-variance for twitter experiments",
# )





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
    supervision_ratio = None
    bias = None
    seed = None
    
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
        elif id_ == "r":
            supervision_ratio = val
        elif id_ == "b":
            bias = val
        elif id_ == "s":
            seed = val
        else:
            raise ValueError(f"Unrecognized ID {id_}.")

    return debiasing, model, experiment, assignment, supervision_ratio, bias, seed

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
    
def _pretty_score(row, tpr_before, tpr_after):
    
    if row[tpr_before] == row[tpr_after]:
        return f"{row[tpr_after]:.3f}"
    elif row[tpr_after] > row[tpr_before]:
        return (
            r"\ua{"
            + f"{row[tpr_after] - row[tpr_before]:.3f}"
            + r"} "
            + f"{row[tpr_after]:.3f}"
        )
    else:
        return (
            r"\da{"
            + f"{row[tpr_before] - row[tpr_after]:.3f}"
            + r"} "
            + f"{row[tpr_after]:.3f}"
        )



if __name__ == "__main__":
    args = parser.parse_args()
        
    results = []
    for file_path in glob.glob(f"{args.tpr_gap_results_dir}/*.json"):

        with open(file_path, "r") as f:
            results.extend(json.load(f))

    records = []
    for experiment in results:
        experiment_id = experiment["experiment_id"]
        
        mean_squared_error_before = experiment["mean_squared_error_before"]
        mean_absolute_error_before = experiment["mean_absolute_error_before"]
        mean_squared_error_after = experiment["mean_squared_error_after"]
        mean_absolute_error_after = experiment["mean_absolute_error_after"]

        MSE_age_before = experiment["MSE_age_before"]
        MAE_age_before = experiment["MAE_age_before"]
        MSE_gender_before = experiment["MSE_gender_before"]
        MAE_gender_before = experiment["MAE_gender_before"]

        MSE_age_after = experiment["MSE_age_after"]
        MAE_age_after = experiment["MAE_age_after"]
        MSE_gender_after = experiment["MSE_gender_after"]
        MAE_gender_after = experiment["MAE_gender_after"]

        assignment_accuracy = experiment["assignment_accuracy"]

        debiasing, model, experiment, assignment, supervision_ratio, bias, seed = _parse_experiment_id(experiment["experiment_id"])

        records.append(
            {
                "experiment_id": experiment_id,
                "model": model,
                "experiment": experiment,
                "debiasing": debiasing,
                "assignment": assignment,
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
                "assignment_accuracy": assignment_accuracy,
                "supervision_ratio": supervision_ratio,
                "seed": seed,
            }
        )
        
    df = pd.DataFrame.from_records(records)

    # Filter to subset of results.
    df = df[df["model"] == args.model]

    # Get pretty model name.
    df["pretty_model_name"] = df.apply(lambda row: _pretty_model_name(row), axis=1)

    # Get pretty metric values.
    df["pretty_mean_squared_error"] = df.apply(lambda row: _pretty_score(row, 'mean_squared_error_before', 'mean_squared_error_after'), axis=1)
    df["pretty_mean_absolute_error"] = df.apply(lambda row: _pretty_score(row, 'mean_absolute_error_before', 'mean_absolute_error_after'), axis=1)

    # Get pretty metric values.
    df["pretty_MSE-age_value"] = df.apply(lambda row: _pretty_score(row, 'MSE_age_before', 'MSE_age_after'), axis=1)
    df["pretty_MAE-age_value"] = df.apply(lambda row: _pretty_score(row, 'MAE_age_before', 'MAE_age_after'), axis=1)

    df["pretty_MSE-gender_value"] = df.apply(lambda row: _pretty_score(row, 'MSE_gender_before', 'MSE_gender_after'), axis=1)
    df["pretty_MAE-gender_value"] = df.apply(lambda row: _pretty_score(row, 'MAE_gender_before', 'MAE_gender_after'), axis=1)

    # To get proper ordering.
    df = df.sort_values(by=["debiasing", "supervision_ratio"])

    df.reset_index(drop = True, inplace = True)

    # "debiased_accuracy": f"{df.iloc[0]['biased_accuracy']:.4f}",
    df_original_model = pd.DataFrame({'pretty_model_name': r"\textsc{" + f"{args.model}" + r"}",
                        'pretty_mean_squared_error': f"{df.iloc[0]['mean_squared_error_before']:.3f}",
                        'pretty_mean_absolute_error': f"{df.iloc[0]['mean_absolute_error_before']:.3f}",
                        'pretty_MSE-age_value': f"{df.iloc[0]['MSE_age_before']:.3f}",
                        'pretty_MAE-age_value': f"{df.iloc[0]['MAE_age_before']:.3f}",
                        'pretty_MSE-gender_value': f"{df.iloc[0]['MSE_gender_before']:.3f}",
                        'pretty_MAE-gender_value': f"{df.iloc[0]['MAE_gender_before']:.3f}",
                        "supervision_ratio": "null"},
                        index=[0])

    df = pd.concat([df_original_model, df])

    # "supervision_ratio",
    # "seed"
    # "pretty_assignment_acc_value",
    
    with pd.option_context("max_colwidth", 1000):
        print(
            df.to_latex(
                float_format="%.3f",
                columns=[
                    "pretty_model_name",
                    "supervision_ratio",
                    "assignment_accuracy",
                    "pretty_mean_squared_error",
                    "pretty_mean_absolute_error",
                    "pretty_MSE-age_value",
                    "pretty_MAE-age_value",
                    "pretty_MSE-gender_value",
                    "pretty_MAE-gender_value"
                ],
                index=False,
                escape=False,
            )
        )

    # os.makedirs(f"{args.persistent_dir}/src/assignment/tables", exist_ok=True)

    with pd.option_context("max_colwidth", 1000):
        with open(
            f"{args.save_path}",
            "w",
        ) as f:
            f.write(
                df.to_latex(
                    float_format="%.3f",
                    columns=[
                        "pretty_model_name",
                        "supervision_ratio",
                        "assignment_accuracy",
                        "pretty_mean_squared_error",
                        "pretty_mean_absolute_error",
                        "pretty_MSE-age_value",
                        "pretty_MAE-age_value",
                        "pretty_MSE-gender_value",
                        "pretty_MAE-gender_value"
                    ],
                    index=False,
                    escape=False,
                )
            )


    if os.path.exists(f"{args.summary_xlsx_path}") == False:
        wb = openpyxl.Workbook()
        wb.save(f"{args.summary_xlsx_path}")

    with pd.ExcelWriter(f"{args.summary_xlsx_path}", engine="openpyxl", mode='a') as writer:  
        df.to_excel(
            writer,
            float_format="%.3f",
            columns=[
                "pretty_model_name",
                "supervision_ratio",
                "assignment_accuracy",
                "pretty_mean_squared_error",
                "pretty_mean_absolute_error",
                "pretty_MSE-age_value",
                "pretty_MAE-age_value",
                "pretty_MSE-gender_value",
                "pretty_MAE-gender_value"
            ],
            sheet_name=f'{args.sheet_name}')