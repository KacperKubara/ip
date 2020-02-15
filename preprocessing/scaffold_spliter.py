import numpy as np
import deepchem
from deepchem.splits import Splitter
from deepchem.utils import ScaffoldGenerator
from deepchem.utils.save import log
from rdkit import Chem


def generate_scaffold(smiles, include_chirality=False):
    """Compute the Bemis-Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    engine = ScaffoldGenerator(include_chirality=include_chirality)
    scaffold = engine.get_scaffold(mol)
    return scaffold


class ScaffoldSplitterNew(Splitter):
    """
    Class for doing data splits based on the scaffold of small molecules.
    """

    def split(self,
              dataset,
              frac_train=.8,
              frac_valid=.1,
              frac_test=.1,
              log_every_n=1000):
        """
        Splits internal compounds into train/validation/test by scaffold.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
        scaffold_sets = self.generate_scaffolds(dataset)

        train_cutoff = frac_train * len(dataset)
        valid_cutoff = (frac_train + frac_valid) * len(dataset)
        train_inds, valid_inds, test_inds = [], [], []

        log("About to sort in scaffold sets", self.verbose)
        for scaffold_set in scaffold_sets:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                    test_inds += scaffold_set
                else:
                    valid_inds += scaffold_set
            else:
                train_inds += scaffold_set
        return train_inds, valid_inds, test_inds

    def generate_scaffolds(self, dataset, log_every_n=1000):
        """
        Returns all scaffolds from the dataset
        """
        scaffolds = {}
        data_len = len(dataset)

        log("About to generate scaffolds", self.verbose)
        for ind, smiles in enumerate(dataset.ids):
            if ind % log_every_n == 0:
                log(f"Generating scaffold {ind} {data_len}", self.verbose)
            scaffold = generate_scaffold(smiles)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)

        # Sort from largest to smallest scaffold sets
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set
            for (scaffold, scaffold_set) in sorted(scaffolds.items(),
                                                   key=lambda x: (len(x[1]), x[1][0]),
                                                   reverse=True)
        ]
        return scaffold_sets


# Testing the class
if __name__ == "__main__":
    splitter = ScaffoldSplitterNew()
    featurizer = "Weave"
    delaney_tasks, delaney_dataset, delaney_transformers = \
        deepchem.molnet.load_delaney(featurizer)
    delaney_train, delaney_valid, delaney_test = delaney_dataset

    scaffold_sets = splitter.generate_scaffolds(delaney_valid)
    print(delaney_valid.ids)
    for i, scaffold in enumerate(scaffold_sets):
        print(f"Scaffold {i}: {scaffold}")
        print(f"Top 3 SMILES entry: {delaney_valid.ids[scaffold[:3]]}\n")
