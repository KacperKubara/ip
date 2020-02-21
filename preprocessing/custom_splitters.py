import numpy as np
import deepchem
from deepchem.splits import Splitter
from deepchem.utils import ScaffoldGenerator
from deepchem.utils.save import log
from rdkit import Chem
from rdkit.Chem import AllChem


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


class MolecularWeightSplitterNew(Splitter):
    """
    Class for doing data splits by molecular weight.
    """
    def split(self,
                dataset,
                seed=None,
                frac_train=.8,
                frac_valid=.1,
                frac_test=.1,
                log_every_n=None):
        """
        Splits internal compounds into train/validation/test using the MW calculated
        by SMILES string.
        """

        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
        self.mws = []
        if not seed is None:
            np.random.seed(seed)

        for smiles in dataset.ids:
            mol = Chem.MolFromSmiles(smiles)
            mw = Chem.rdMolDescriptors.CalcExactMolWt(mol)
            self.mws.append(mw)

        # Sort by increasing MW
        self.mws = np.array(self.mws)
        self.sortidx = np.argsort(self.mws)

        train_cutoff = int(frac_train * len(self.sortidx))
        valid_cutoff = int((frac_train + frac_valid) * len(self.sortidx))
        print(f"train_cutoff: {train_cutoff}, valid_cutoff: {valid_cutoff}")
        return (self.sortidx[:train_cutoff], self.sortidx[train_cutoff:valid_cutoff],
                self.sortidx[valid_cutoff:])


def ClusterFps(fps, cutoff=0.2):
    # (ytz): this is directly copypasta'd from Greg Landrum's clustering example.
    dists = []
    nfps = len(fps)
    from rdkit import DataStructs
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    from rdkit.ML.Cluster import Butina
    cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    return cs

class ButinaSplitterNew(Splitter):
    """
    Class for doing data splits based on the butina clustering of a bulk tanimoto
    fingerprint matrix.
    """

    def split(self,
              dataset,
              seed=None,
              frac_train=None,
              frac_valid=None,
              frac_test=None,
              log_every_n=1000,
              cutoff=0.18,
              regression_task=True):
        """
        Splits internal compounds into train and validation based on the butina
        clustering algorithm. This splitting algorithm has an O(N^2) run time, where N
        is the number of elements in the dataset. The dataset is expected to be a classification
        dataset.
        This algorithm is designed to generate validation data that are novel chemotypes.
        Note that this function entirely disregards the ratios for frac_train, frac_valid,
        and frac_test. Furthermore, it does not generate a test set, only a train and valid set.
        Setting a small cutoff value will generate smaller, finer clusters of high similarity,
        whereas setting a large cutoff value will generate larger, coarser clusters of low similarity.
        """
        print("Performing butina clustering with cutoff of", cutoff)
        scaffold_sets = self.generate_scaffolds(dataset, cutoff)
        ys = dataset.y
        valid_inds = []

        for c_idx, cluster in enumerate(scaffold_sets):
            # for m_idx in cluster:
            valid_inds.extend(cluster)
            # continue until we find an active in all the tasks, otherwise we can't
            # compute a meaningful AUC
            # TODO (ytz): really, we want at least one active and inactive in both scenarios.
            # TODO (Ytz): for regression tasks we'd stop after only one cluster.
            active_populations = np.sum(ys[valid_inds], axis=0)
            if np.all(active_populations):
                print("# of actives per task in valid:", active_populations)
                print("Total # of validation points:", len(valid_inds))
                break

        train_inds = list(itertools.chain.from_iterable(scaffold_sets[c_idx + 1:]))
        test_inds = []

        return train_inds, valid_inds, []

    def generate_scaffolds(self, dataset, cutoff=0.18):
        """
        Returns all scaffolds from the dataset 
        """
        mols = []
        for ind, smiles in enumerate(dataset.ids):
            mols.append(Chem.MolFromSmiles(smiles))
        n_mols = len(mols)
        fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]

        scaffold_sets = ClusterFps(fps, cutoff=cutoff)
        scaffold_sets = sorted(scaffold_sets, key=lambda x: -len(x))
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
