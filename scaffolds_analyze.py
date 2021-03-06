""" Script to analyze statistical properties of the scaffold """
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from yellowbrick.cluster import KElbowVisualizer
import deepchem as dc

from preprocessing import ScaffoldSplitterNew, MolecularWeightSplitterNew, ButinaSplitterNew
from data import load_wang_data_gcn
from utils import generate_scaffold_metrics

path_results = "/home/kjk1u17/ip/ip/results/wang_dataset"
splitter_dict ={
    "Butina": ButinaSplitterNew(),
    "Scaffold": ScaffoldSplitterNew(), 
    }

if __name__ == "__main__":
    wang_tasks, wang_dataset, wang_transformers =\
        load_wang_data_gcn(featurizer="GraphConv", split='index', frac_train=0.99,
                           frac_test=0.005, frac_valid=0.005)
    wang_train, wang_valid, wang_test = wang_dataset
    
    for name, splitter in splitter_dict.items():
        if name == "Butina":
            results = [list(), list(), list(), list()]
            for i in range (0, 10, 100):
                print(f"Generating Butina scaffold: {i}")
                i_float = float(i)/100
                scaffold_sets = splitter.generate_scaffolds(wang_train, cutoff=i_float)
                scaffold_dict_sum = {k: len(v) for k, v in enumerate(scaffold_sets)}
                scaffold_sum = [v for v in scaffold_dict_sum.values()]
                more_than_5 =  len([v for v in scaffold_sum if v >= 5 and v < 50])
                more_than_50 =  len([v for v in scaffold_sum if v >= 50 and v <= 100])
                more_than_100 =  len([v for v in scaffold_sum if v >= 100])

                results[0].append(i_float)
                results[1].append(more_than_5)
                results[2].append(more_than_50)
                results[3].append(more_than_100)
            
            results_dict = {}
            results_dict["x"] = deepcopy(results[0])
            results_dict["Count"] = results[1]
            results_dict["Type"] = ["50> count >=5" for i in range(len(results[1]))]

            results_dict["x"] += deepcopy(results[0])
            results_dict["Count"] += results[2]
            results_dict["Type"] += ["100> count >=50" for i in range(len(results[2]))]

            results_dict["x"] += deepcopy(results[0])
            results_dict["Count"] += results[3]
            results_dict["Type"] += ["count >=100" for i in range(len(results[3]))]
            
            print(results_dict)
            sc_plot = sns.scatterplot(x="x", y="Count", hue="Type", data=results_dict)
            plt.title(f'Butina Sweep: Count of cluster sizes for specific Tanimoto threshold')
            plt.xlabel('Tanimoto threshold')
            plt.ylabel('Count of clusters')
            sc_plot.figure.savefig(path_results + f"/Butina_sweep.png")
            plt.clf()

            scaffold_sets = splitter.generate_scaffolds(wang_train, cutoff=0.8)
        else:
            scaffold_sets = splitter.generate_scaffolds(wang_train)

        scaffold_dict = {k: v for k, v in enumerate(scaffold_sets)}
        scaffold_dict_sum = {k: len(v) for k, v in enumerate(scaffold_sets)}
        scaffold_dict_sum = {k: v for k, v in
                            sorted(scaffold_dict_sum.items(),
                            key=lambda item: item[1], reverse=True)
                            }
        scaffold_sum = [v for v in scaffold_dict_sum.values()]
        scaffold_unique = list(set([v for v in scaffold_dict_sum.values()]))
        print(f"{name}: Raw Scaffolds: {scaffold_sets}")
        print(f"{name}: Raw Scaffolds Length: {[len(sfd) for sfd in scaffold_sets]}")
        print(f"{name}: Scaffolds sorted and unique: {scaffold_unique}")
        scaffold_unique.sort(reverse=True)

        # Plotting unique set of scaffold sizes
        plt.tight_layout()
        plt.figure(figsize=(12, 8))
        bar_plot = sns.barplot(x=[i for i in range(len(scaffold_unique))],
                            y=scaffold_unique)
        plt.title(f'{name} model: All possible Scaffold Sizes')
        plt.xlabel('Scaffold ID')
        plt.ylabel('Scaffold Size')
        bar_plot.figure.savefig(path_results + f"/{name}_unique.png")
        plt.clf()

        # Plotting distribution of scaffold sizes
        hist_plot = sns.distplot(scaffold_sum, kde=False)
        plt.title(f'{name} model: Histogram of Scaffold Sizes with log scale')
        plt.xlabel('Scaffold Size')
        plt.ylabel('Count of scaffold size (log scale)')
        plt.yscale('log')
        hist_plot.figure.savefig(path_results + f"/{name}_dist.png")
        plt.clf()

        # Plotting distribution of scaffold sizes bigger than 5
        scaffold_sum_threshold = [v for v in scaffold_sum if v > 5]
        hist_plot = sns.distplot(scaffold_sum_threshold, kde=False)
        plt.title(f'{name} model: Histogram of Scaffold Sizes that are bigger than 5')
        plt.xlabel('Scaffold Size')
        plt.ylabel('Count of scaffold size')
        hist_plot.figure.savefig(path_results + f"/{name}_dist_thresholded.png")
        plt.clf()


    # Clustering data based on the molecular weight
    splitter_mw = MolecularWeightSplitterNew()
    splitter_mw.split(wang_train)
    print(f"splitter.mws: {splitter_mw.mws}")
    print(f"splitter.sortidx: {splitter_mw.sortidx}")

    kmeans = MiniBatchKMeans(n_clusters=3,
                                random_state=44,
                                batch_size=6,
                                max_iter=100)
    visualizer = KElbowVisualizer(kmeans, k=(1,12))
    mws = np.array(splitter_mw.mws).reshape(-1, 1)
    visualizer.fit(mws)
    visualizer.show(outpath=path_results + "/mws_elbow.png")
    plt.clf()

    # Visualize the weight clusters
    kmeans = MiniBatchKMeans(n_clusters=3,
                            random_state=44,
                            batch_size=6,
                            max_iter=100)
    labels = kmeans.fit_predict(mws).flatten()
    df = pd.DataFrame({"mws": mws.flatten(), "labels": labels})
    hist_plot = sns.distplot(df[labels==0]["mws"], color="red", kde=False, label="Heavy")
    sns.distplot(df[labels==2]["mws"], color="olive", ax=hist_plot, kde=False, label="Medium")
    sns.distplot(df[labels==1]["mws"], color="skyblue", ax=hist_plot, kde=False, label="Light")
    plt.title(f'Clustered distribution of molecular weights with MiniBatchKMeans, n_clusters=3')
    plt.xlabel('Molecular Weights')
    plt.ylabel('Count')
    plt.legend()
    hist_plot.figure.savefig(path_results + f"/MolecularWeightDist_separate.png")
    plt.clf()

    model_dict = {
        "n_clusters": 3, 
        "random_state": 44, 
        "batch_size": 6, 
        "max_iter": 10
        }
    scaffold_sets = splitter_mw.generate_scaffolds(MiniBatchKMeans, model_dict)
    print(scaffold_sets)
    print(len(scaffold_sets))
    print(f"Heavy molecule count: {len(df[labels==0].index)}")
    print(f"Medium molecule count: {len(df[labels==2].index)}")
    print(f"Light molecule count: {len(df[labels==1].index)}")
    for arr in scaffold_sets:
        print(f"Length of the array: {len(arr)}")
