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
        elif id_ == "u":
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



def _pretty_accuracy_value(row):
    
    if row['biased_accuracy'] == row['debiased_accuracy']:
        # return r"\ua{" + f"{row['debiased_accuracy']:.2f}" + r"}"
        return f"{row['debiased_accuracy']:.2f}"
    elif row['debiased_accuracy'] > row['biased_accuracy']:
        return (
            r"\uag{"
            + f"{row['debiased_accuracy'] - row['biased_accuracy']:.2f}"
            + r"} "
            + f"{row['debiased_accuracy']:.2f}"
        )
    else:
        return (
            r"\dab{"
            + f"{row['biased_accuracy'] - row['debiased_accuracy']:.2f}"
            + r"} "
            + f"{row['debiased_accuracy']:.2f}"
        )


def _pretty_tpr_value_unweighted(row):
    
    if row['tpr_gap_before_unweighted'] == row['tpr_gap_after_unweighted']:
        return f"{row['tpr_gap_after_unweighted']:.2f}"
    elif row['tpr_gap_after_unweighted'] > row['tpr_gap_before_unweighted']:
        return (
            r"\ua{"
            + f"{row['tpr_gap_after_unweighted'] - row['tpr_gap_before_unweighted']:.2f}"
            + r"} "
            + f"{row['tpr_gap_after_unweighted']:.2f}"
        )
    else:
        return (
            r"\da{"
            + f"{row['tpr_gap_before_unweighted'] - row['tpr_gap_after_unweighted']:.2f}"
            + r"} "
            + f"{row['tpr_gap_after_unweighted']:.2f}"
        )

def _pretty_tpr_value(row):
    
    if row['tpr_gap_before'] == row['tpr_gap_after']:
        # return r"\dab{" + f"{row['tpr_gap_after']:.2f}" + r"}"
        return f"{row['tpr_gap_after']:.2f}"
    elif row['tpr_gap_after'] > row['tpr_gap_before']:
        return (
            r"\ua{"
            + f"{row['tpr_gap_after'] - row['tpr_gap_before']:.2f}"
            + r"} "
            + f"{row['tpr_gap_after']:.2f}"
        )
    else:
        return (
            r"\da{"
            + f"{row['tpr_gap_before'] - row['tpr_gap_after']:.2f}"
            + r"} "
            + f"{row['tpr_gap_after']:.2f}"
        )
        
def _pretty_f1_macro_value(row):
    
    if row['f1_macro_before'] == row['f1_macro_after']:
        return f"{row['f1_macro_after']:.4f}"
    elif row['f1_macro_after'] > row['f1_macro_before']:
        return (
            r"\uag{"
            + f"{row['f1_macro_after'] - row['f1_macro_before']:.4f}"
            + r"} "
            + f"{row['f1_macro_after']:.4f}"
        )
    else:
        return (
            r"\dab{"
            + f"{row['f1_macro_before'] - row['f1_macro_after']:.4f}"
            + r"} "
            + f"{row['f1_macro_after']:.4f}"
        )


def _pretty_f1_micro_value(row):
    
    if row['f1_micro_before'] == row['f1_micro_after']:
        # return r"\dab{" + f"{row['tpr_gap_after']:.2f}" + r"}"
        return f"{row['f1_micro_after']:.4f}"
    elif row['f1_micro_after'] > row['f1_micro_before']:
        return (
            r"\uag{"
            + f"{row['f1_micro_after'] - row['f1_micro_before']:.4f}"
            + r"} "
            + f"{row['f1_micro_after']:.4f}"
        )
    else:
        return (
            r"\dab{"
            + f"{row['f1_micro_before'] - row['f1_micro_after']:.4f}"
            + r"} "
            + f"{row['f1_micro_after']:.4f}"
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
        biased_accuracy = experiment["biased_accuracy"]
        debiased_accuracy = experiment["debiased_accuracy"]
        tpr_gap_before = experiment["tpr_gap_before"]
        tpr_gap_after = experiment["tpr_gap_after"]
        tpr_gap_before_unweighted = experiment["tpr_gap_before_unweighted"]
        tpr_gap_after_unweighted = experiment["tpr_gap_after_unweighted"]
        f1_macro_before = experiment['f1_macro_before']
        f1_micro_before = experiment['f1_micro_before']
        f1_macro_after = experiment['f1_macro_after']
        f1_micro_after = experiment['f1_micro_after']
        assignment_accuracy = str(experiment["assignment_accuracy"])

        debiasing, model, experiment, assignment, supervision_ratio, bias, seed = _parse_experiment_id(experiment["experiment_id"])

        records.append(
            {
                "experiment_id": experiment_id,
                "model": model,
                "experiment": experiment,
                "debiasing": debiasing,
                "assignment": assignment,
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
                "assignment_accuracy": assignment_accuracy,
                "supervision_ratio": supervision_ratio,
            }
        )
        
                # "assignment_accuracy": assignment_accuracy,
                # "supervision_ratio": supervision_ratio,
                # "seed": seed,
    df = pd.DataFrame.from_records(records)

    # Filter to subset of results.
    df = df[df["model"] == args.model]

    # Get pretty model name.
    df["pretty_model_name"] = df.apply(lambda row: _pretty_model_name(row), axis=1)

    # Get pretty metric values.
    df["pretty_acc_value"] = df.apply(lambda row: _pretty_accuracy_value(row), axis=1)

    # Get pretty metric values.
    df["pretty_tpr-gap_value"] = df.apply(lambda row: _pretty_tpr_value(row), axis=1)

    df["pretty_tpr-gap_value_unweighted"] = df.apply(lambda row: _pretty_tpr_value_unweighted(row), axis=1)
    
    df["pretty_f1_macro_value"] = df.apply(lambda row: _pretty_f1_macro_value(row), axis=1)

    df["pretty_f1_micro_value"] = df.apply(lambda row: _pretty_f1_micro_value(row), axis=1)


    # To get proper ordering.
    df = df.sort_values(by=["debiasing", "supervision_ratio"])
    
    df.reset_index(drop = True, inplace = True)

    # "debiased_accuracy": f"{df.iloc[0]['biased_accuracy']:.4f}",
    df_original_model = pd.DataFrame({'pretty_model_name': r"\textsc{" + f"{args.model}" + r"}",
                        'pretty_acc_value': f"{df.iloc[0]['biased_accuracy']:.4f}",
                        'pretty_tpr-gap_value': f"{df.iloc[0]['tpr_gap_before']:.6f}",
                        'pretty_tpr-gap_value_unweighted': f"{df.iloc[0]['tpr_gap_before_unweighted']:.2f}",
                        "pretty_f1_macro_value": f"{df.iloc[0]['f1_macro_before']:.2f}",
                        "pretty_f1_micro_value":  f"{df.iloc[0]['f1_micro_before']:.2f}",
                        "supervision_ratio": "null",
                        "assignment_accuracy": "null"},
                        index=[0])


    # INLP_debiased_accuracy = df[df['debiasing'] == "INLP"]['debiased_accuracy'].mean()
    # INLP_tpr_gap_after = df[df['debiasing'] == "INLP"]['tpr_gap_after'].mean()
    # INLP_tpr_gap_after_unweighted = df[df['debiasing'] == "INLP"]['tpr_gap_after_unweighted'].mean()
    # INLP_f1_macro_after = df[df['debiasing'] == "INLP"]['f1_macro_after'].mean()
    # INLP_f1_micro_after = df[df['debiasing'] == "INLP"]['f1_micro_after'].mean()
    # SAL_debiased_accuracy = df[df['debiasing'] == "SAL"]['debiased_accuracy'].mean()
    # SAL_tpr_gap_after = df[df['debiasing'] == "SAL"]['tpr_gap_after'].mean()
    # SAL_tpr_gap_after_unweighted = df[df['debiasing'] == "SAL"]['tpr_gap_after_unweighted'].mean()
    # SAL_f1_macro_after = df[df['debiasing'] == "SAL"]['f1_macro_after'].mean()
    # SAL_f1_micro_after = df[df['debiasing'] == "SAL"]['f1_micro_after'].mean()

    # df_SAL_mean = pd.DataFrame({'pretty_model_name': r"\, + \textsc{SR} + \textsc{SAL}",
    #                     'biased_accuracy': df.iloc[0]['biased_accuracy'],
    #                     'debiased_accuracy': SAL_debiased_accuracy,
    #                     'tpr_gap_before': df.iloc[0]['tpr_gap_before'],
    #                     'tpr_gap_after': SAL_tpr_gap_after,
    #                     'tpr_gap_before_unweighted': df.iloc[0]['tpr_gap_before_unweighted'],
    #                     'tpr_gap_after_unweighted': SAL_tpr_gap_after_unweighted,
    #                     'f1_macro_before': df.iloc[0]['f1_macro_before'],
    #                     'f1_macro_after': SAL_f1_macro_after,
    #                     'f1_micro_before': df.iloc[0]['f1_micro_before'],
    #                     'f1_micro_after': SAL_f1_micro_after},
    #                     index=[0])

    # df_SAL_mean["pretty_acc_value"] = df_SAL_mean.apply(lambda row: _pretty_accuracy_value(row), axis=1)
    # df_SAL_mean["pretty_tpr-gap_value"] = df_SAL_mean.apply(lambda row: _pretty_tpr_value(row), axis=1)
    # df_SAL_mean["pretty_tpr-gap_value_unweighted"] = df_SAL_mean.apply(lambda row: _pretty_tpr_value_unweighted(row), axis=1)
    # df_SAL_mean["pretty_f1_macro_value"] = df_SAL_mean.apply(lambda row: _pretty_f1_macro_value(row), axis=1)
    # df_SAL_mean["pretty_f1_micro_value"] = df_SAL_mean.apply(lambda row: _pretty_f1_micro_value(row), axis=1)


    # df_INLP_mean = pd.DataFrame({'pretty_model_name': r"\, + \textsc{SR} + \textsc{INLP}",
    #                     'biased_accuracy': df.iloc[0]['biased_accuracy'],
    #                     'debiased_accuracy': INLP_debiased_accuracy,
    #                     'tpr_gap_before': df.iloc[0]['tpr_gap_before'],
    #                     'tpr_gap_after': INLP_tpr_gap_after,
    #                     'tpr_gap_before_unweighted': df.iloc[0]['tpr_gap_before_unweighted'],
    #                     'tpr_gap_after_unweighted': INLP_tpr_gap_after_unweighted,
    #                     'f1_macro_before': df.iloc[0]['f1_macro_before'],
    #                     'f1_macro_after': INLP_f1_macro_after,
    #                     'f1_micro_before': df.iloc[0]['f1_micro_before'],
    #                     'f1_micro_after': INLP_f1_micro_after},
    #                     index=[0])

    # df_INLP_mean["pretty_acc_value"] = df_INLP_mean.apply(lambda row: _pretty_accuracy_value(row), axis=1)
    # df_INLP_mean["pretty_tpr-gap_value"] = df_INLP_mean.apply(lambda row: _pretty_tpr_value(row), axis=1)
    # df_INLP_mean["pretty_tpr-gap_value_unweighted"] = df_INLP_mean.apply(lambda row: _pretty_tpr_value_unweighted(row), axis=1)
    # df_INLP_mean["pretty_f1_macro_value"] = df_INLP_mean.apply(lambda row: _pretty_f1_macro_value(row), axis=1)
    # df_INLP_mean["pretty_f1_micro_value"] = df_INLP_mean.apply(lambda row: _pretty_f1_micro_value(row), axis=1)


    # df = pd.concat([df_original_model, df, df_SAL_mean, df_INLP_mean])

    df = pd.concat([df_original_model, df])

    with pd.option_context("max_colwidth", 1000):
        print(
            df.to_latex(
                float_format="%.3f",
                columns=[
                    "pretty_model_name",
                    "supervision_ratio",
                    "assignment_accuracy",
                    "pretty_acc_value",
                    "pretty_tpr-gap_value",
                    "pretty_tpr-gap_value_unweighted",
                    "pretty_f1_macro_value",
                    "pretty_f1_micro_value",
                ],
                index=False,
                escape=False,
            )
        )
        
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
                        "pretty_acc_value",
                        "pretty_tpr-gap_value",
                        "pretty_tpr-gap_value_unweighted",
                        "pretty_f1_macro_value",
                        "pretty_f1_micro_value",
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
                    "pretty_acc_value",
                    "pretty_tpr-gap_value",
                    "pretty_tpr-gap_value_unweighted",
                    "pretty_f1_macro_value",
                    "pretty_f1_micro_value",
            ],
            sheet_name=f'{args.sheet_name}')
