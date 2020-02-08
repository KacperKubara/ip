""" Utility functions to load the data"""
import logging

import pandas as pd 
import deepchem
from deepchem.data import CSVLoader
from deepchem.feat import ConvMolFeaturizer, WeaveFeaturizer, CircularFingerprint
from deepchem.splits import ButinaSplitter, ScaffoldSplitter, MolecularWeightSplitter, MaxMinSplitter, IndexSplitter
from deepchem.trans import NormalizationTransformer
from deepchem.models import GraphConvModel, WeaveModel, MPNNModel

from config import PATH_SOL_DATA, PATH_WANG_DATA_SMILES

logger = logging.getLogger(__name__)

def load_sol_challenge() -> pd.DataFrame:
    """ Loads Solubility Challenge dataset"""
    logger.info("About to load Solubility challenge data")
    return pd.read_csv(PATH_SOL_DATA)

def load_wang_data() -> pd.DataFrame:
    """ Loads Wang dataset"""
    logger.info("About to load Wang data")
    return pd.read_csv(PATH_WANG_DATA_SMILES)

def load_wang_data_gcn(featurizer='GraphConv', split='index', move_mean=True,
                       frac_train=0.8, frac_valid=0.1, frac_test=0.1) -> pd.DataFrame:
    """ Loads Wang dataset to utilize it with GCNs from deepchem"""
    logger.info("About to load and featurize Wang dataset")
    wang_tasks = ['ClogP']

    # Choosing featurizers
    if featurizer == 'ECFP':
        featurizer = deepchem.feat.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
        featurizer = deepchem.feat.ConvMolFeaturizer()
    elif featurizer == 'Weave':
        featurizer = deepchem.feat.WeaveFeaturizer()

    loader = deepchem.data.CSVLoader(
        tasks=wang_tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(PATH_WANG_DATA_SMILES, shard_size=8192)

    if split is None:
        transformers = [
            deepchem.trans.NormalizationTransformer(
                transform_y=True, dataset=dataset, move_mean=move_mean)
        ]

        logger.info("Split is None, about to transform data")
        for transformer in transformers:
            dataset = transformer.transform(dataset)

        return wang_tasks, (dataset, None, None), transformers

    # Splitting data
    splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter(),
      'stratified': deepchem.splits.SingletaskStratifiedSplitter()
    }
    splitter = splitters[split]
    logger.info("About to split dataset with {} splitter.".format(split))
    train, valid, test = splitter.train_valid_test_split(dataset,
                                                         frac_train=frac_train,
                                                         frac_valid=frac_valid,
                                                         frac_test=frac_test)

    transformers = [
        deepchem.trans.NormalizationTransformer(
            transform_y=True, dataset=train, move_mean=move_mean)
    ]

    logger.info("About to transform data.")
    for transformer in transformers:
        train = transformer.transform(train)
        valid = transformer.transform(valid)
        test = transformer.transform(test)

    return wang_tasks, (train, valid, test), transformers
