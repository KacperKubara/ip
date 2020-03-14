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
                          tensorboard_log_frequency=25)
    else:
        model = model_obj(len(wang_tasks),
                          batch_size=50,
                          mode='regression',
                          tensorboard_log_frequency=25)
    return model

splitter_dict = {
    "Random": RandomSplitter(),
    "Scaffold": ScaffoldSplitterNew(),
    "MolecularWeight": MolecularWeightSplitterNew(),
    "Butina": ButinaSplitterNew(),
    }


if __name__ == "__main__":
    results = {}

    for splitter_name, splitter in splitter_dict.items():
        logging.info(f"Generating scaffolds with {splitter_name}")
        results[splitter_name] = {}
        for model_name, model_obj in model_dict.items():
            logging.info(f"Using {model_name} as a model")
            results[splitter_name][model_name] = {}
            featurizer = model_name
            wang_tasks, wang_dataset, wang_transformers =\
                load_wang_data_gcn(featurizer, split='index', frac_train=0.99,
                                   frac_test=0.005, frac_valid=0.005)
            wang_train, wang_valid, wang_test = wang_dataset
            metric = dc.metrics.Metric(dc.metrics.mae_score)
            splitter_rand = RandomSplitter() # For CV

            # Get the biggest scaffolds
            if splitter_name == "Butina":
                scaffold_sets = splitter.generate_scaffolds(wang_train, cutoff=0.8)
            if splitter_name == "MolecularWeight":
                splitter.split(wang_train)
                scaffold_sets = splitter.generate_scaffolds(MiniBatchKMeans, kmeans_dict)
            if splitter_name == "Scaffold":
                scaffold_sets = splitter.generate_scaffolds(wang_train)
            if splitter_name == "Random":
                scaffold_sets = splitter.split(wang_train, frac_train=0.32,
                                               frac_valid=0.33, frac_test=0.35)

            logging.info(f"Scaffolds sets size: {len(scaffold_sets)}")
            logging.info(f"Scaffolds length: {[len(sfd) for sfd in scaffold_sets]}")
            logging.info(f"Raw scaffolds: {scaffold_sets}")
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
            with open(path_base + ".json", 'w') as outfile:    
                json.dump(results, outfile)
                logging.info("Succesful save to json file")