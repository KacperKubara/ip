# Script implementing the 1st benchmark from my IP project
# ToDo: Reimplement scaffold to give all clusters separetely
import os
import json
import logging
logging.getLogger().setLevel(logging.INFO)

import numpy as np
import deepchem as dc
from deepchem.models import GraphConvModel, WeaveModel
from deepchem.models import MultitaskRegressor
from deepchem.models import ValidationCallback
from rdkit import Chem

from data import load_wang_data_gcn
from preprocessing import ScaffoldSplitterNew

# GCN models used for the benchmark
model_dict = {
    "GraphConv": GraphConvModel,
    "Weave": WeaveModel,
    "ECFP": MultitaskRegressor,
}
splitter_methods = ["random"]
path_results = "./results/benchmark1_2/results_benchmark1_2.json"


def generate_scaffold_metrics(model, data_valid, metric, transformers):
    results = {}
    splitter = ScaffoldSplitterNew()
    scaffold_sets = splitter.generate_scaffolds(data_valid)

    for i, scaffold in enumerate(scaffold_sets):
        # Taking a subset of data is possible only in this way
        print(f"Scaffold {i} size: {len(scaffold)}")
        data_subset = data_valid.select(indices=scaffold)
        valid_scores = model.evaluate(data_subset, [metric], transformers)
        results[f"Scaffold_{i}"] = {}
        results[f"Scaffold_{i}"]["results"] = valid_scores
        results[f"Scaffold_{i}"]["results"]["logP"] = data_subset.y.ravel().tolist()
        results[f"Scaffold_{i}"]["results"]["logP_mean"] = np.mean(data_subset.y).ravel().tolist()
        results[f"Scaffold_{i}"]["results"]["logP_std"] = np.std(data_subset.y).ravel().tolist()
        results[f"Scaffold_{i}"]["smiles"] = data_valid.ids[scaffold].tolist()
    return results


if __name__ == "__main__":
    results = {}
    for splitter in splitter_methods:
        results[splitter] = {}
        for model_name, model_obj in model_dict.items():
            featurizer = model_name

            wang_tasks, wang_dataset, wang_transformers =\
                load_wang_data_gcn(featurizer, split=splitter)
            wang_train, wang_valid, wang_test = wang_dataset
            train_mols = [Chem.MolFromSmiles(compounds)
                          for compounds in wang_train.ids]
            logging.info(f"Size of the training data: {len(wang_train.ids)}")
            logging.info(f"Size of the test data: {len(wang_test.ids)}")
            logging.info(f"Size of the validation data: {len(wang_valid.ids)}")

            if model_name == "ECFP":
                model = model_obj(len(wang_tasks),
                                  wang_train.get_data_shape()[0],
                                  batch_size=50,
                                  model_dir=f"./results/benchmark1_2/tensorboard_logs/{splitter}/{model_name}",
                                  tensorboard=True,
                                  tensorboard_log_frequency=25)

            else:
                model = model_obj(len(wang_tasks),
                                  batch_size=50,
                                  mode='regression',
                                  model_dir=f"./results/benchmark1_2/tensorboard_logs/{splitter}/{model_name}",
                                  tensorboard=True,
                                  tensorboard_log_frequency=25)
            # Pearson metric won't be applicable for small scaffolds!
            # For smaller size scaffolds use rms
            metric = dc.metrics.Metric(dc.metrics.mae_score)
            # After 25 batch updates, measure the loss
            loss_logger = ValidationCallback(wang_valid,
                                             interval=25,
                                             metrics=[metric])

            logging.info(f"Fitting the model: {model_name}")
            model.fit(wang_train, nb_epoch=10)

            train_scores = model.evaluate(wang_train,
                                          [metric],
                                          wang_transformers)
            valid_scores = generate_scaffold_metrics(model,
                                                     wang_valid,
                                                     metric,
                                                     wang_transformers)

            results[splitter][model_name] = {}
            results[splitter][model_name]["train_score"] = train_scores
            results[splitter][model_name]["valid_score"] = valid_scores

    logging.info(results)
    logging.info(f"Results has been saved to {path_results}")
    with open(path_results, 'w') as json_f:
        json.dump(results, json_f)
