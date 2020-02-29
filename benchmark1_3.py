# Do the Scaffold Split
# Take the k (2) biggest scaffolds
# Generate 10 different folds with RandomSplitter
# Get the scores with the uncertainty range 
import json
import logging
logging.getLogger().setLevel(logging.INFO)

import deepchem as dc
from deepchem.models import GraphConvModel, WeaveModel
from deepchem.models import MultitaskRegressor
from deepchem.splits import RandomSplitter

from preprocessing import ScaffoldSplitterNew, ButinaSplitterNew, MolecularWeightSplitterNew
from data import load_wang_data_gcn
from sklearn.cluster import MiniBatchKMeans

# Dict with params for KMeans
kmeans_dict = {
    "n_clusters": 3, 
    "random_state": 44, 
    "batch_size": 6, 
    "max_iter": 10
    }

# GCN models used for the benchmark
model_dict = {
    "GraphConv": GraphConvModel,
    "Weave": WeaveModel,
    "ECFP": MultitaskRegressor,
    }
path_base = "./results/benchmark1_3/results_benchmark1_3_"

def get_model(model_name: str):
    if model_name == "ECFP":
        model = model_obj(len(wang_tasks),
                          wang_train.get_data_shape()[0],
                          batch_size=50,
                          model_dir=f"./results/benchmark1_3/tensorboard_logs/{splitter}/{model_name}",
                          tensorboard=True,
                          tensorboard_log_frequency=25)
    else:
        model = model_obj(len(wang_tasks),
                          batch_size=50,
                          mode='regression',
                          model_dir=f"./results/benchmark1_3/tensorboard_logs/{splitter}/{model_name}",
                          tensorboard=True,
                          tensorboard_log_frequency=25)
    return model

splitter_dict ={
    "MolecularWeight": MolecularWeightSplitterNew(),
    "Butina": ButinaSplitterNew(),
    "Scaffold": ScaffoldSplitterNew(), 
    }


if __name__ == "__main__":
    results = {}

    for splitter_name, splitter in splitter_dict.items():
        results[splitter_name] = {}
        for model_name, model_obj in model_dict.items():
            results[splitter_name][model_name] = {}
            featurizer = model_name
            wang_tasks, wang_dataset, wang_transformers =\
                load_wang_data_gcn(featurizer, split='index', frac_train=0.99,
                                frac_test=0.005, frac_valid=0.005)
            wang_train, wang_valid, wang_test = wang_dataset
            metric = dc.metrics.Metric(dc.metrics.mae_score)
            splitter_rand = RandomSplitter() # For CV

            # Get two biggest scaffolds
            if splitter_name == "Butina":
                scaffold_sets = splitter.generate_scaffolds(wang_train, cutoff=0.8)
            if splitter_name == "MolecularWeight":
                splitter.split(wang_train)
                scaffold_sets = splitter.generate_scaffolds(MiniBatchKMeans, kmeans_dict)
            else:
                scaffold_sets = splitter.generate_scaffolds(wang_train)
            logging.info(f"Scaffolds sets size: {len(scaffold_sets)}")
            scaffold_sets_filt = [sfd for sfd in scaffold_sets if len(sfd) >= 100]
        
            for sfd_filt in scaffold_sets_filt:
                sfd_name = "scaffold_" + str(len(sfd_filt))
                results[splitter_name][model_name][sfd_name] = {}
                logging.info(f"Scaffold size: {len(sfd_filt)}")

                data_subset = wang_train.select(indices=sfd_filt)
                k_fold = splitter_rand.k_fold_split(data_subset, k=10)
                for i, fold in enumerate(k_fold):
                    model = get_model(model_name)
                    train, valid = fold
                    logging.info(f"Train size: {len(train)}, Valid size: {len(valid)}")
                    
                    model.fit(train)
                    train_scores = model.evaluate(train,
                                                [metric],
                                                wang_transformers)
                    valid_scores = model.evaluate(valid,
                                                [metric],
                                                wang_transformers)

                    fold_name = "fold_" + str(i)
                    results[splitter_name][model_name][sfd_name][fold_name] = {}
                    results[splitter_name][model_name][sfd_name][fold_name]["train score"] = train_scores
                    results[splitter_name][model_name][sfd_name][fold_name]["valid score"] = valid_scores
                    # Make sure to use a new model each time
                    del model

            # Update results file after each model
            with open(path_results + splitter_name + ".json", 'w') as outfile:    
                json.dump(results, outfile)
                logging.info("Succesful save to json file")