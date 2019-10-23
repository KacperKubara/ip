""" Encodes SMILES to Extended Connectivity Finger Prints
or to Graphs.
"""
import pandas as pd 
from rdkit import Chem
from rdkit.Chem import AllChem

class SMILESEncoder():
    """ Encodes SMILES to vector, or matrix-
    like format used for CNN
    
    Parameters
    ----------
    col_name: [str]
        Column name which contains SMILES string
    """
    def __init__(self, col_name):
        self.col_name = col_name

    def to_ecfp(self, X: pd.DataFrame, bit_vector: bool = True) -> pd.Series:
        """ Runs specific EDA model
        
        Parameters
        ----------
        X : pandas DataFrame
            Data with SMILES column
        
        bit_vector: bool
            Whether to use Morgan Bit Vector Fingerprint

        Returns
        -------
        pandas Series:
            ECFP transformed from SMILES
        """
        ss_mols = \
            X[self.col_name].apply(lambda x: Chem.MolFromSmiles(x))
        if bit_vector == True:
            ss_ecfp = \
                ss_mols.apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, radius=2))
        else:
            ss_ecfp = \
                ss_mols.apply(lambda x: AllChem.GetMorganFingerprint(x, radius=2))
                
        return ss_ecfp

    def to_graphs(self):
        pass