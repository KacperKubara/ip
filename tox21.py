import numpy as np 
import tensorflow as tf
import deepchem as dc
from deepchem.models import GraphConvModel
import pandas as pd
from IPython.display import Image, display

tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = tox21_datasets

model = GraphConvModel(len(tox21_tasks), batch_size=50, mode='classification', 
                            tensorboard=True, model_dir="./tensorboard_logs")
# Set nb_epoch=10 for better results.
model.fit(train_dataset, nb_epoch=5)

metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, np.mean, mode="classification")

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
print("Training ROC-AUC Score: %f" % train_scores["mean-roc_auc_score"])
valid_scores = model.evaluate(valid_dataset, [metric], transformers)
print("Validation ROC-AUC Score: %f" % valid_scores["mean-roc_auc_score"])