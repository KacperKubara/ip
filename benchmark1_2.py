# Script implementing the 1st benchmark from my IP project
import json
import logging
from copy import deepcopy
logging.getLogger().setLevel(logging.INFO)

import numpy as np
import tensorflow as tf
import deepchem as dc
from deepchem.models import GraphConvModel, WeaveModel
from deepchem.models import MultitaskRegressor
from deepchem.hyper import HyperparamOpt
from rdkit import Chem

from data import load_wang_data_gcn
from utils import generate_scaffold_metrics

# GCN models used for the benchmark
model_dict = {
    "GraphConv": GraphConvModel,
    "Weave": WeaveModel,
    "ECFP": MultitaskRegressor,
}
params_dict = {
    "GraphConv": {
        "graph_conv_layers": [[32, 32], [64, 64], [128, 128]],
        "dense_layer_size": [128, 256, 512],
        "dropout": [0.0, 0.2],
        "mode": ["regression"]
    },
    "Weave": {
        "n_hidden": [32, 64, 128],
        "mode": ["regression"]
    },
    "ECFP": {
        "layer_sizes": [[1000], [1000, 1000], [1000, 1000, 1000]],
        "dropouts": [0.0, 0.2, 0.5],
        "activation_fns": [tf.nn.leaky_relu]
    }
}
params_save = {}
splitter_methods = ["random"]
path_results = "./results/benchmark1_2/results_benchmark1_2.json"
path_best_params = "./results/benchmark1_2/results_benchmark1_2_params.json"

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
                params_model = params_dict[model_name]
                params_model["n_tasks"] = [len(wang_tasks)]
                params_model["n_features"] = [wang_train.get_data_shape()[0]]
                params_model["batch_size"] = [50]
                
            if model_name == "GraphConv":
                params_model = params_dict[model_name]
                params_model["n_tasks"] = [len(wang_tasks)]
                params_model["batch_size"] = [50]
                
            if model_name == "Weave":
                params_model = params_dict[model_name]
                params_model["n_tasks"] = [len(wang_tasks)]
                params_model["batch_size"] = [50]

            def model_builder(model_params, model_dir):
                return model_obj(**model_params, model_dir=model_dir)

            # Pearson metric won't be applicable for small scaffolds!
            # For smaller size scaffolds use rms
            metric = dc.metrics.Metric(dc.metrics.mae_score)
            opt = HyperparamOpt(model_builder)
            params_best, score_best, all_results = opt.hyperparam_search(params_model, 
                                                                        wang_train,
                                                                        wang_valid,
                                                                        wang_transformers,
                                                                        metric)
            
            logging.info(f"Best score for {model_name}: {score_best}")
            logging.info(f"All results for {model_name}: {all_results}")
            logging.info(f"Best params for {model_name}: {params_best}")
            params_save[model_name] = deepcopy(params_best)
            model = model_obj(**params_best)
            
            logging.info(f"Fitting the best model model: {model_name}")
            model.fit(wang_train, nb_epoch=10)

            train_scores = model.evaluate(wang_train,
                                          [metric],
                                          wang_transformers)
            valid_scores = generate_scaffold_metrics(model,
                                                     wang_valid,
                                                     metric,
                                                     wang_transformers,
                                                     logdir=None)

            results[splitter][model_name] = {}
            results[splitter][model_name]["train_score"] = train_scores
            results[splitter][model_name]["valid_score"] = valid_scores

        logging.info(results)
        logging.info(f"Results has been saved to {path_results}")
        with open(path_results, 'w') as json_f:
            json.dump(results, json_f)
        with open(path_best_params, 'w') as json_f:
            json.dump(params_best, json_f)