# Script to analyze statistical properties of the scaffold
import numpy as np
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
                                max_iter=10)
    visualizer = KElbowVisualizer(kmeans, k=(1,12))
    mws = np.array(splitter_mw.mws).reshape(-1, 1)
    visualizer.fit(mws)
    visualizer.show(outpath=path_results + "/mws_elbow.png")

    model_dict = {
        "n_clusters": 3, 
        "random_state": 44, 
        "batch_size": 6, 
        "max_iter": 10
        }
    scaffold_sets = splitter_mw.generate_scaffolds(MiniBatchKMeans, model_dict)
    print(scaffold_sets)
    print(len(scaffold_sets))
    for arr in scaffold_sets:
        print(f"Lengthof the array: {len(arr)}")
