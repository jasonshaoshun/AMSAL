import argparse
import glob
import json
import os
import re

import pandas as pd
from os.path import exists as file_exists

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Export CrowS-Pairs results.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--bias_type",
    action="store",
    type=str,
    default="gender",
    choices=["gender", "race", "religion"],
    help="Type of bias to export results for.",
)
parser.add_argument(
    "--model_type",
    action="store",
    type=str,
    choices=["bert", "albert", "roberta", "gpt2"],
    help="What model type to export results for.",
)


def _label_model_type(row):
    if "Bert" in row["model"]:
        return "bert"
    elif "Albert" in row["model"]:
        return "albert"
    elif "Roberta" in row["model"]:
        return "roberta"
    else:
        return "gpt2"


def _pretty_model_name(row):
    pretty_name_mapping = {
        "BertForMaskedLM": "BERT",
        "SentenceDebiasBertForMaskedLM": r"\, + \textsc{SentenceDebias}",
        "INLPBertForMaskedLM": r"\textsc{INLP}",
        "CDABertForMaskedLM": r"\, + \textsc{CDA}",
        "DropoutBertForMaskedLM": r"\, + \textsc{Dropout}",
        "SelfDebiasBertForMaskedLM": r"\, + \textsc{Self-Debias}",
        "AlbertForMaskedLM": "ALBERT",
        "SentenceDebiasAlbertForMaskedLM": r"\, + \textsc{SentenceDebias}",
        "INLPAlbertForMaskedLM": r"\textsc{INLP}",
        "CDAAlbertForMaskedLM": r"\, + \textsc{CDA}",
        "DropoutAlbertForMaskedLM": r"\, + \textsc{Dropout}",
        "SelfDebiasAlbertForMaskedLM": r"\, + \textsc{Self-Debias}",
        "RobertaForMaskedLM": "RoBERTa",
        "SentenceDebiasRobertaForMaskedLM": r"\, + \textsc{SentenceDebias}",
        "INLPRobertaForMaskedLM": r"\textsc{INLP}",
        "CDARobertaForMaskedLM": r"\, + \textsc{CDA}",
        "DropoutRobertaForMaskedLM": r"\, + \textsc{Dropout}",
        "SelfDebiasRobertaForMaskedLM": r"\, + \textsc{Self-Debias}",
        "GPT2LMHeadModel": "GPT-2",
        "SentenceDebiasGPT2LMHeadModel": r"\, + \textsc{SentenceDebias}",
        "INLPGPT2LMHeadModel": r"\textsc{INLP}",
        "CDAGPT2LMHeadModel": r"\, + \textsc{CDA}",
        "DropoutGPT2LMHeadModel": r"\, + \textsc{Dropout}",
        "SelfDebiasGPT2LMHeadModel": r"\, + \textsc{Self-Debias}",
        "SALBertForMaskedLM": r"\textsc{SAL}",
        "SALAlbertForMaskedLM": r"\textsc{SAL}",
        "SALRobertaForMaskedLM": r"\textsc{SAL}",
        "SALGPT2LMHeadModel": r"\textsc{SAL}",
    }

    pretty_assignment_mapping = {
        "Sal": r"\, + \textsc{AM} + ",
        "Kmeans": r"\, + \textsc{Kmeans} + ",
        "Oracle": r"\, + \textsc{Oracle}",
        "partialSup": r"\, + \textsc{Partial}",
    }

    pretty_name = pretty_assignment_mapping[row["pre_assignment"]] + pretty_name_mapping[row["model"]]
    
    if pretty_name == r"\, + \textsc{AM} + \textsc{SAL}":
        pretty_name = "\, + \textsc{AM}\textsc{SAL}"

    return pretty_name

def _get_baseline_metric(df, model_type):
    model_type_to_baseline = {
        "bert": "BertForMaskedLM",
        "albert": "AlbertForMaskedLM",
        "roberta": "RobertaForMaskedLM",
        "gpt2": "GPT2LMHeadModel",
    }
    baseline = model_type_to_baseline[model_type]
    return df[df["model"] == baseline]["metric"].values[0]


def _pretty_metric_value(row, baseline_metric):
    baseline_diff = abs(baseline_metric - 50)
    debias_diff = abs(row["metric"] - 50)

    if debias_diff == baseline_diff:
        return f"{row['metric']:.2f}"
    elif debias_diff < baseline_diff:
        return (
            r"\da{"
            + f"{baseline_diff - debias_diff:.2f}"
            + r"} "
            + f"{row['metric']:.2f}"
        )
    else:
        return (
            r"\ua{"
            + f"{debias_diff - baseline_diff:.2f}"
            + r"} "
            + f"{row['metric']:.2f}"
        )


def _parse_experiment_id(experiment_id):
    model = None
    model_name_or_path = None
    bias_type = None
    seed = None

    items = experiment_id.split("_")[1:]
    for item in items:
        id_, val = item[:1], item[2:]
        if id_ == "m":
            model = val
        elif id_ == "c":
            model_name_or_path = val
        elif id_ == "t":
            bias_type = val
        elif id_ == "s":
            seed = int(val)
        elif id_ == "p":
            pre_assignment = val
        else:
            raise ValueError(f"Unrecognized ID {id_}.")

    return model, model_name_or_path, bias_type, seed, pre_assignment


if __name__ == "__main__":
    args = parser.parse_args()

    print("Exporting CrowS-Pairs results:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - bias_type: {args.bias_type}")

    records = []
    for file_path in glob.glob(f"{args.persistent_dir}/results/crows/*.json"):
        # Parse the experiment ID.
        file_name = os.path.basename(file_path)
        experiment_id = os.path.splitext(file_name)[0]
        model, model_name_or_path, bias_type, seed, pre_assignment = _parse_experiment_id(experiment_id)

        # Load the results.
        with open(file_path, "r") as f:
            metric = json.load(f)

        records.append(
            {
                "experiment_id": experiment_id,
                "model": model,
                "model_name_or_path": model_name_or_path,
                "bias_type": bias_type,
                "seed": seed,
                "metric": metric,
                "pre_assignment": pre_assignment,
            }
        )

    df = pd.DataFrame.from_records(records)

    # print(f"original df is\n")
    # with pd.option_context("max_colwidth", 1000):
    #     print(
    #         df.to_latex(
    #             float_format="%.2f",
    #             index=False,
    #             escape=False,
    #         )
    #     )

    # Label model type (e.g., "bert").
    df["model_type"] = df.apply(lambda row: _label_model_type(row), axis=1)

    # Get pretty model name.
    df["pretty_model_name"] = df.apply(lambda row: _pretty_model_name(row), axis=1)

    # Filter to subset of results.
    df = df[(df["model_type"] == args.model_type) & (df["bias_type"] == args.bias_type)]
    # baseline_metric = df["metric"].values[0]
    baseline_metric = _get_baseline_metric(df, args.model_type)

    # Get pretty metric values.
    df["pretty_metric_value"] = df.apply(
        lambda row: _pretty_metric_value(row, baseline_metric), axis=1
    )

    # Only include results for the first seed.
    df = df[(df["seed"] == 0) | (df["seed"].isnull())]

    # To get proper ordering.
    df = df.sort_values(by="pretty_model_name")
    
    # if file_exists(f"{args.persistent_dir}/results/crows/all.xlsx"):
    #     with pd.ExcelWriter(f"{args.persistent_dir}/results/crows/all.xlsx", mode='a', if_sheet_exists='overlay') as writer:
    #         df.to_excel(
    #             writer,
    #             header=False,
    #             float_format="%.2f",
    #             columns=["model_type", "model", "pre_assignment", "bias_type", "metric"],
    #             index=False,
    #             sheet_name="Sheet2",
    #         )
    # else:
    #     df.to_excel(
    #         f"{args.persistent_dir}/results/crows/all.xlsx",
    #         float_format="%.2f",
    #         columns=["model_type", "model", "pre_assignment", "bias_type", "metric"],
    #         index=False,
    #         sheet_name="Sheet2",
    #     )



    # if file_exists(f"{args.persistent_dir}/results/crows/all.xlsx"):
    #     with pd.ExcelWriter(f"{args.persistent_dir}/results/crows/all.xlsx", mode='a') as writer:
    #         df.to_excel(
    #             writer,
    #             float_format="%.2f",
    #             columns=["model_type", "model", "pre_assignment", "bias_type", "metric"],
    #             index=False,
    #             sheet_name=f"{args.model_type}_{args.bias_type}",
    #         )
    # else:
    #     df.to_excel(
    #         f"{args.persistent_dir}/results/crows/all.xlsx",
    #         float_format="%.2f",
    #         columns=["model_type", "model", "pre_assignment", "bias_type", "metric"],
    #         index=False,
    #         sheet_name=f"{args.model_type}_{args.bias_type}",
    #     )
    df.reset_index(drop = True, inplace = True)
    df = df.reindex([4, 1, 2, 5, 7, 0, 3, 6, 8])


    with pd.option_context("max_colwidth", 1000):
        print(
            df.to_latex(
                float_format="%.2f",
                columns=["pretty_model_name", "pretty_metric_value"],
                index=False,
                escape=False,
            )
        )

    os.makedirs(f"{args.persistent_dir}/tables", exist_ok=True)
    with pd.option_context("max_colwidth", 1000):
        with open(
            f"{args.persistent_dir}/tables/crows_m-{args.model_type}_t-{args.bias_type}.tex",
            "w",
        ) as f:
            f.write(
                df.to_latex(
                    float_format="%.2f",
                    columns=["pretty_model_name", "pretty_metric_value"],
                    index=False,
                    escape=False,
                )
            )
