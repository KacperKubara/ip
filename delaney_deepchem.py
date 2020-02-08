# Script to test if all Graph models work correctly
import os

import pandas as pd 
import numpy as np
import tensorflow as tf
import deepchem as dc
from deepchem.models import GraphConvModel, WeaveModel, MPNNModel
from rdkit import Chem

model_dict = {
    "MPNN": MPNNModel,
    "GraphConv": GraphConvModel,
    "Weave": WeaveModel,
}

if __name__ == "__main__":
    results = {}
    for model_name, model_obj in model_dict.items():
        featurizer = model_name
        if featurizer == "MPNN":
            featurizer = "Weave"
        delaney_tasks, delaney_dataset, delaney_transformers = dc.molnet.load_delaney(featurizer)
        delaney_train, delaney_valid, delaney_test = delaney_dataset
        train_mols = [Chem.MolFromSmiles(compounds)
                    for compounds in delaney_train.ids]
        if model_name == "MPNN":
            model = model_obj(
                len(delaney_tasks),
                n_atom_feat=75,
                n_pair_feat=14,
                T=3,
                M=5,
                batch_size=64,
                learning_rate=0.0001,
                use_queue=False,
                mode='regression',
            )
        else:
            model = model_obj(
            len(delaney_tasks),
            batch_size=50,
            mode='regression',
        )
        
        metric = dc.metrics.Metric(dc.metrics.pearson_r2_score) # Same metric for all models
        
        model.fit(delaney_train, nb_epoch=10)
        train_scores = model.evaluate(delaney_train, [metric], delaney_transformers) # Metrics are evaluated over the whole data set -> not ideal
        valid_scores = model.evaluate(delaney_valid, [metric], delaney_transformers)
        
        print(f"Train Scores: {model_name}")
        print(train_scores)
        print(f"Validation Scores: {model_name}")
        print(valid_scores)
        results[model_name] = (train_scores, valid_scores)
    
    print(results)