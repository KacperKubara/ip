from IPython.display import Image, display
import deepchem as dc
from deepchem.molnet import load_tox21
from deepchem.models import GraphConvModel

# Load Tox21 dataset
tox21_tasks, tox21_datasets, transformers = load_tox21(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = tox21_datasets
# Construct the model with tensorbaord on
model = GraphConvModel(len(tox21_tasks),  
                       mode='classification',
                       tensorboard=True)

# Fit the model
model.fit(train_dataset, nb_epoch=10)