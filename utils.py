""" Utility functions for general, package-wide purposes"""
import logging
import json
from copy import deepcopy

import numpy as np
import pandas as pd
from rdkit import Chem
import seaborn as sns
import matplotlib.pyplot as plt

from preprocessing import ScaffoldSplitterNew, ButinaSplitterNew, MolecularWeightSplitterNew
from sklearn.cluster import MiniBatchKMeans


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
            for splitter_name, dict_split in dict_valid.items():
                tuple_sorted = sorted(dict_split.items(),
                                      key=lambda t: t[1]["results"]["mae_score"])
                dict_sorted = {k: v for k, v in tuple_sorted}
                results_dict[splitter][model]["valid_score"][splitter_name] = dict_sorted

    with open(path_write, 'w') as json_f:
        json.dump(results_dict, json_f)


def visualize_benchmark1_3_results(path_read: str = None, path_write: str = None):
    results_dict = {}
    with open(path_read, 'r') as json_f:
        results_dict = json.load(json_f)

    data_plot = {}
    for splitter_name, dict0 in results_dict.items():
        data_plot[splitter_name] = {}
        for model_name, dict1 in dict0.items():
            data_plot[splitter_name][model_name] = {}
            for scaffold_name, dict2 in dict1.items():
                data_plot[splitter_name][model_name][scaffold_name] = {}
                arr_temp = []
                for fold_name, dict3 in dict2.items():
                    arr_temp.append(dict3["valid score"]["mae_score"])
                data_plot[splitter_name][model_name][scaffold_name]["valid_mae"] = arr_temp

    # Save the data for preliminary insights
    with open(path_write, 'w') as json_f:
        json.dump(data_plot, json_f)

    df_converter_temp = {
        "model": [], 
        "scaffold": [],
        "mae": []
        }
    df_converter = {}
    for splitter_name, dict0 in data_plot.items():
        df_converter_temp["model"] = []
        df_converter_temp["scaffold"] = []
        df_converter_temp["mae"] = []
        for model_name, dict1 in dict0.items():
            for scaffold, dict2 in dict1.items():
                for el in dict2["valid_mae"]:
                    df_converter_temp["model"].append(model_name)
                    df_converter_temp["scaffold"].append(scaffold)
                    df_converter_temp["mae"].append(el)
        df_converter[splitter_name] = deepcopy(df_converter_temp)

    for splitter_name, df_converter_temp in df_converter.items():
        print(splitter_name)
        df = pd.DataFrame(df_converter_temp)
        print(df["scaffold"].unique())
        ax = sns.boxplot(x="model", y="mae", hue="scaffold", 
                        data=df_converter_temp, palette="Set3")
        ax.set(xlabel='Models', ylabel='MAE', ylim=(0, 1.4))
        ax.set_title(f'MAE with different models on scaffolded data ({splitter_name})')
        ax.figure.savefig(path_write[:-4] + f"_{splitter_name}_boxplots.png")
        plt.clf()

def generate_scaffold_metrics(model, data_valid, metric, transformers):
    results = {}
    splitters = {
        "Scaffold": ScaffoldSplitterNew(), 
        "Butina": ButinaSplitterNew(), 
        "MolecularWeight": MolecularWeightSplitterNew()
        }
    for splitter_name, splitter in splitters.items():
        results[splitter_name] = {}
        if splitter_name == "Butina":   
            scaffold_sets = splitter.generate_scaffolds(data_valid, cutoff=0.8)
        if splitter_name == "MolecularWeight":
            splitter.split(data_valid)
            # Dict with params for KMeans
            kmeans_dict = {
            "n_clusters": 3, 
            "random_state": 44, 
            "batch_size": 6, 
            "max_iter": 10
            }
            scaffold_sets = splitter.generate_scaffolds(MiniBatchKMeans, kmeans_dict)
        if splitter_name == "Scaffold":
            scaffold_sets = splitter.generate_scaffolds(data_valid)
        for i, scaffold in enumerate(scaffold_sets):
            # Taking a subset of data is possible only in this way
            print(f"Scaffold {i} size: {len(scaffold)}")
            if len(scaffold) < 5:
                print(f"Scaffold {i} is smaller than 5. Skipping...")
                continue
            data_subset = data_valid.select(indices=scaffold)
            y_rescaled = data_subset.y

            # Untransfoming data has to be done in the reversed order
            for transformer in reversed(transformers):
                y_rescaled = transformer.untransform(y_rescaled)
            y_rescaled = y_rescaled.ravel().tolist()

            valid_scores = model.evaluate(data_subset, [metric], transformers)
            logp_pred = model.predict(data_subset, transformer)
            results[splitter_name][f"Scaffold_{i}"] = {}
            results[splitter_name][f"Scaffold_{i}"]["results"] = valid_scores
            results[splitter_name][f"Scaffold_{i}"]["results"]["logP_true"] = y_rescaled
            results[splitter_name][f"Scaffold_{i}"]["results"]["logP_pred"] = logp_pred
            results[splitter_name][f"Scaffold_{i}"]["results"]["logP_mean"] = float(np.mean(y_rescaled))
            results[splitter_name][f"Scaffold_{i}"]["results"]["logP_std"] = float(np.std(y_rescaled))
            results[splitter_name][f"Scaffold_{i}"]["smiles"] = data_valid.ids[scaffold].tolist()
    return results

if __name__ == "__main__":
    """    
    sort_benchmark1_results("./results/benchmark1_2/results_benchmark1_2.json",
                            "./results/benchmark1_2/results_benchmark1_2_sorted.json")
    """

    visualize_benchmark1_3_results("./results/benchmark1_3/results_benchmark1_3_all.json",
                                   "./results/benchmark1_3/results_benchmark1_3_for_plots.json")