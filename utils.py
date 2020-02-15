""" Utility functions for general, package-wide purposes"""
import logging

import numpy as np
from deepchem.utils import ScaffoldGenerator
from deepchem.splits import Splitter
from rdkit import Chem


def average(x: list) -> float:
    """ Averages all elements in the list
    
    Parameters
    ----------
    x : [int, float]
        list with int or float elements
    
    Returns
    -------
    float, np.nan:
        Returns float value when averaging is successful.
        np.nan otherwise
    """
    try:
        average = sum(x)/float(len(x))
    except (TypeError, ValueError):
        average = np.nan

    return average


class ScaffoldSplitter(Splitter):
    """
    Class for doing data splits based on the scaffold of small molecules.
    """

    def split(self,
              dataset,
              seed=None,
              frac_train=.8,
              frac_valid=.1,
              frac_test=.1,
              log_every_n=1000):
        """
        Splits internal compounds into train/validation/test by scaffold.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
        scaffolds = {}
        logging.info("About to generate scaffolds", self.verbose)
        data_len = len(dataset)
        for ind, smiles in enumerate(dataset.ids):
            if ind % log_every_n == 0:
                logging.info("Generating scaffold %d/%d" % (ind, data_len), self.verbose)
            scaffold = self.generate_scaffold(smiles)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)
        # Sort from largest to smallest scaffold sets
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]
        train_cutoff = frac_train * len(dataset)
        valid_cutoff = (frac_train + frac_valid) * len(dataset)
        train_inds, valid_inds, test_inds = [], [], []
        logging.info("About to sort in scaffold sets", self.verbose)
        for scaffold_set in scaffold_sets:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                    test_inds += scaffold_set
                else:
                    valid_inds += scaffold_set
            else:
                train_inds += scaffold_set
        return train_inds, valid_inds, test_inds

    def generate_scaffold(self, smiles, include_chirality=False):
        """ Compute the Bemis-Murcko scaffold for a SMILES string"""
        mol = Chem.MolFromSmiles(smiles)
        engine = ScaffoldGenerator(include_chirality=include_chirality)
        scaffold = engine.get_scaffold(mol)
        return scaffold
