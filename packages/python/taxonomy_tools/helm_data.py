import os
import numpy as np
from typing import List
import json

# This structure contains the translation of the taxonomy dataset strings
# into actual HELM dataset results.
# An entry in this dictionary does not need to be exactly one dataset, nor be
# constituted of the same dataset, arbitrary collections can be passed. Only
# keep in mind that the metrics should all be of the same type, since all
# results within a dataset entry will be averaged.
dataset_config = {
    "LegalSupport": [
        {
            "name": "legal_support,method=multiple_choice_joint",  # Partial string of the result name, all the characters before ",model=..."
            "metric": "exact_match",  # Type of metric to look for in the "stats.json" file. The first match is the one used.
            "field": "mean",  # Field of the metric to keep, usually mean.
            "split": "test",  # Split of the metric, some dataset have "test" split, other only have "validation"
        }
    ],
    "Synthetic_reasoning_(natural_language)": [
        {
            "name": "synthetic_reasoning_natural:difficulty=hard",
            "metric": "f1_set_match",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "synthetic_reasoning_natural:difficulty=easy",
            "metric": "f1_set_match",
            "field": "mean",
            "split": "test",
        },
    ],
    "Synthetic_reasoning_(abstract_symbols)---pattern_match": [
        {
            "name": "synthetic_reasoning:mode=pattern_match",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "Synthetic_reasoning_(abstract_symbols)---variable_sustitution": [
        {
            "name": "synthetic_reasoning:mode=variable_substitution",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "Synthetic_reasoning_(abstract_symbols)---induction": [
        {
            "name": "synthetic_reasoning:mode=induction",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "bAbI": [
        {
            "name": "babi_qa:task=all",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "LSAT": [
        {
            "name": "lsat_qa:task=all",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "HellaSwag": [
        {
            "name": "commonsense:dataset=hellaswag",
            "metric": "exact_match",
            "field": "mean",
            "split": "valid",
        }
    ],
    "OpenBookQA": [
        {
            "name": "commonsense:dataset=openbookqa",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "MMLU": [
        {
            "name": "mmlu:subject=abstract_algebra",
            "metric": "exact_match",
            "field": "mean",
            "split": "valid",
        },
        {
            "name": "mmlu:subject=college_chemistry",
            "metric": "exact_match",
            "field": "mean",
            "split": "valid",
        },
        {
            "name": "mmlu:subject=computer_security",
            "metric": "exact_match",
            "field": "mean",
            "split": "valid",
        },
        {
            "name": "mmlu:subject=econometrics",
            "metric": "exact_match",
            "field": "mean",
            "split": "valid",
        },
        {
            "name": "mmlu:subject=us_foreign_policy",
            "metric": "exact_match",
            "field": "mean",
            "split": "valid",
        },
    ],
    "WikiText-103": [
        {
            "name": "IGNORE-ME",
        }
    ],
    "The Pile": [
        {
            "name": "the_pile:subset=ArXiv",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "the_pile:subset=BookCorpus2",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "the_pile:subset=Enron Emails",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "the_pile:subset=Github",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "the_pile:subset=PubMed Central",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "the_pile:subset=Wikipedia (en)",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
    ],
    "TwitterAAE": [
        {
            "name": "twitter_aae:demographic=white",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "twitter_aae:demographic=aa",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
    ],
    "ICE": [
        {
            "name": "ice:gender=female",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "ice:gender=male",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "ice:subset=ea",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "ice:subset=hk",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "ice:subset=ind",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "ice:subset=usa",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
    ],
    "WikiData": [
        {
            "name": "IGNORE-ME",
        }
    ],
    "BLiMP": [
        {
            "name": "blimp:phenomenon=binding,method=multiple_choice_separate_original",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "blimp:phenomenon=irregular_forms,method=multiple_choice_separate_original",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "blimp:phenomenon=island_effects,method=multiple_choice_separate_original",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "blimp:phenomenon=quantifiers,method=multiple_choice_separate_original",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        },
    ],
}


def read_helm_data(
    helm_data_path: str,
    datasets_list: List[str],
    verbose: bool = False,
    current_dict: dict = {},
) -> dict:
    """
    Given a dataset list from a taxonomy, loads all metrics from the HELM
    results directory.
    Returns a dictionary containing a key for each dataset and within each key
    a dictionary of models and their respective metrics values.
    """

    helm_samples_dict = current_dict

    # Get a list of tasks results from the helm dir
    tasks_dirs = os.listdir(helm_data_path)

    for dataset in datasets_list:
        if dataset in helm_samples_dict.keys():
            print(
                "Skipping dataset %s, it is already present in the provided samples dictionary."
                % dataset
            )
            continue

        # Get list of associated tasks
        task_list = dataset_config[dataset]
        tasks_results_dict = dict()
        ignored = False
        for task in task_list:
            if task["name"] == "IGNORE-ME":
                print("Ignoring dataset : %s" % dataset)
                ignored = True
                continue
            # Get all results matching this task
            matching_results = [
                t_dir
                for t_dir in tasks_dirs
                if (task["name"] + "," in t_dir and "groups" not in t_dir)
            ]
            if len(matching_results) == 0:
                if verbose:
                    print("task not found : %s" % task["name"])

            # Get the tested models names
            tested_models = [
                result_name.split("model=")[-1].split(",")[0]
                for result_name in matching_results
            ]
            assert len(tested_models) == len(np.unique(tested_models))

            # Get the metrics for each model
            for this_result, this_model in zip(matching_results, tested_models):
                metric_val = None
                with open(os.path.join(helm_data_path, this_result, "stats.json")) as f:
                    task_results = json.load(f)
                for r in task_results:
                    try:
                        if (
                            r["name"]["name"] == task["metric"]
                            and r["name"]["split"] == task["split"]
                        ):
                            metric_val = r[task["field"]]

                            if task["metric"] == "bits_per_byte":
                                # just try to set this in the same range as accuracy...
                                metric_val /= 8
                                metric_val = 1.0 - metric_val
                            # The first is the one we want
                            break
                    except Exception:
                        pass
                if metric_val is None:
                    raise ValueError("Requested metric not found.")

                # track all values from the tasks associated to this dataset
                if this_model not in tasks_results_dict.keys():
                    tasks_results_dict[this_model] = list()
                tasks_results_dict[this_model].append(metric_val)

        # Keep only models with all the tasks tested in this dataset and calculate the mean
        tasks_results_dict = {
            k: np.mean(v)
            for k, v in tasks_results_dict.items()
            if len(v) == len(task_list)
        }

        if not ignored:
            if len(tasks_results_dict) > 0:
                helm_samples_dict[dataset] = tasks_results_dict
            else:
                if verbose:
                    print("No data found for dataset: %s" % dataset)

    return helm_samples_dict


def split_helm_result_folder_name(folder_name):
    split_data = dict()
    split_data["dataset"] = folder_name.split(":")[0]
    split_data["model"] = folder_name.split(",model=")[-1].split(",")[0]
    if "task" in folder_name:
        split_data["task"] = folder_name.split(":task=")[-1].split(",")[0]
    return split_data
