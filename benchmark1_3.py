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

from preprocessing import ScaffoldSplitterNew
from data import load_wang_data_gcn

# GCN models used for the benchmark
model_dict = {
    "GraphConv": GraphConvModel,
    "Weave": WeaveModel,
    "ECFP": MultitaskRegressor,
}
path_results = "./results/benchmark1_3/results_benchmark1_3.json"

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

if __name__ == "__main__":
    results = {}

    for model_name, model_obj in model_dict.items():
        results[model_name] = {}
        featurizer = model_name
        wang_tasks, wang_dataset, wang_transformers =\
            load_wang_data_gcn(featurizer, split='index', frac_train=0.99,
                               frac_test=0.005, frac_valid=0.005)
        wang_train, wang_valid, wang_test = wang_dataset
        metric = dc.metrics.Metric(dc.metrics.mae_score)
        splitter = ScaffoldSplitterNew()
        splitter_rand = RandomSplitter() # For CV

        # Get two biggest scaffolds
        scaffold_sets = splitter.generate_scaffolds(wang_train)
        scaffold_sets_filt = [sfd for sfd in scaffold_sets if len(sfd) >= 800]
    
        assert len(scaffold_sets_filt) == 2 # Should be 2 for Wang dataset

        for sfd_filt in scaffold_sets_filt:
            sfd_name = "scaffold_" + str(len(sfd_filt))
            results[model_name][sfd_name] = {}
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
                results[model_name][sfd_name][fold_name] = {}
                results[model_name][sfd_name][fold_name]["train score"] = train_scores
                results[model_name][sfd_name][fold_name]["valid score"] = valid_scores
                # Make sure to use a new model each time
                del model

        # Update results file after each model
        with open(path_results, 'w') as outfile:    
            json.dump(results, outfile)
            logging.info("Succesful save to json file")