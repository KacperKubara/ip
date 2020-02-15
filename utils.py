""" Utility functions for general, package-wide purposes"""
import logging
import json

import numpy as np
from deepchem.utils import ScaffoldGenerator
from deepchem.splits import Splitter
from rdkit import Chem


def average(x: list) -> float:
    """ Averages all elements in the list

    Parameters
    ----------
    x : [int, float]
        list with int or float elements

    Returns
    -------
    float, np.nan:
        Returns float value when averaging is successful.
        np.nan otherwise
    """
    try:
        average = sum(x)/float(len(x))
    except (TypeError, ValueError):
        average = np.nan
    return average


def sort_benchmark1_results(path_read: str, path_write: str):
    results_dict = {}
    with open(path_read, 'r') as json_f:
        results_dict = json.load(json_f)

    for splitter, dict1 in results_dict.items():
        for model, dict2 in dict1.items():
            dict_valid = dict2["valid_score"]
            tuple_sorted = sorted(dict_valid.items(),
                                  key=lambda t: t[1]["results"]["mae_score"])
            dict_sorted = {k: v for k, v in tuple_sorted}
            results_dict[splitter][model]["valid_score"] = dict_sorted

    with open(path_write, 'w') as json_f:
        json.dump(results_dict, json_f)

if __name__ == "__main__":
    sort_benchmark1_results("./results/benchmark1_2/results_benchmark1_2.json",
                            "./results/benchmark1_2/results_benchmark1_2_sorted.json")