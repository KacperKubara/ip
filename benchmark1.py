# Script implementing the 1st benchmark from my IP project
import logging
logging.getLogger().setLevel(logging.INFO)

import deepchem as dc
from deepchem.models import GraphConvModel, WeaveModel
from deepchem.models import MultitaskRegressor
from deepchem.models import ValidationCallback
from rdkit import Chem

from data import load_wang_data_gcn

# GCN models used for the benchmark
model_dict = {
    "GraphConv": GraphConvModel,
    "Weave": WeaveModel,
    "ECFP": MultitaskRegressor,
}

splitter_methods = ["scaffold", "butina", "index"]
path_results = "./results/benchmark1/results_benchmark1.csv"

if __name__ == "__main__":
    results = {}
    for splitter in splitter_methods:
        results[splitter] = {}
        for model_name, model_obj in model_dict.items():
            featurizer = model_name

            wang_tasks, wang_dataset, wang_transformers =\
                load_wang_data_gcn(featurizer, split=splitter)
            wang_train, wang_valid, wang_test = wang_dataset
            if splitter == "butina":
                logging.warn("Butina doesn't return a test set, copying valid set into test set")
                wang_test = wang_valid
            train_mols = [Chem.MolFromSmiles(compounds)
                          for compounds in wang_train.ids]
            logging.info(f"Size of the training data: {len(wang_train.ids)}")
            logging.info(f"Size of the test data: {len(wang_test.ids)}")
            logging.info(f"Size of the validation data: {len(wang_valid.ids)}")

            if model_name == "ECFP":
                model = model_obj(len(wang_tasks),
                                  wang_train.get_data_shape()[0],
                                  batch_size=50,
                                  model_dir=f"./results/benchmark1/tensorboard_logs/{splitter}/{model_name}",
                                  tensorboard=True,
                                  tensorboard_log_frequency=25)

            else:
                model = model_obj(len(wang_tasks),
                                  batch_size=50,
                                  mode='regression',
                                  model_dir=f"./results/benchmark1/tensorboard_logs/{splitter}/{model_name}",
                                  tensorboard=True,
                                  tensorboard_log_frequency=25)

            metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
            # After 25 batch updates, measure the loss
            loss_logger = ValidationCallback(wang_valid,
                                             interval=25,
                                             metrics=[metric])

            logging.info(f"Fitting the model: {model_name}")
            model.fit(wang_train, nb_epoch=10)

            train_scores = model.evaluate(wang_train, [metric], wang_transformers)
            valid_scores = model.evaluate(wang_valid, [metric], wang_transformers)

            print(f"Train Scores for {model_name}")
            print(train_scores)
            print(f"Validation Scores for {model_name}")
            print(valid_scores)
            results[splitter][model_name] = (train_scores, valid_scores)

            with open(path_results, 'w') as f:
                for key in results.keys():
                    f.write(f"{key},{results[key]}\n")

    print(results)
