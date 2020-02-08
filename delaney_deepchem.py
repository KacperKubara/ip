import os

import pandas as pd 
import numpy as np
import tensorflow as tf
import deepchem as dc
from deepchem.models import GraphConvModel
from rdkit import Chem
from sklearn.model_selection import train_test_split

from preprocessing import PreProcessor
from config import PATH_RESULTS, PATH_RESULTS_EDA_CORR,\
    PATH_RESULTS_EDA_DIST

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == "__main__":
    delaney_tasks, delaney_dataset, delaney_transformers = dc.molnet.load_delaney(featurizer='GraphConv')
    delaney_train, delaney_valid, delaney_test = delaney_dataset
    train_mols = [Chem.MolFromSmiles(compounds)
                for compounds in delaney_train.ids]
    model = GraphConvModel(
        len(delaney_tasks),
        batch_size=50,
        mode='regression',
        tensorboard=True,
        model_dir="./tensorboard_logs",
        verbose=1
    )
    model.fit(delaney_train, nb_epoch=10)
    metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score, np.mean, mode="regression"
    )

    train_scores = model.evaluate(delaney_train, [metric], delaney_transformers)
    valid_scores = model.evaluate(delaney_valid, [metric], delaney_transformers)
    print("Validation ROC-AUC Score: %f" % valid_scores["mean-roc_auc_score"])