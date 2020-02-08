import pandas as pd 
import numpy as no
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
    def __init__(self, col_name: str):
        self.col_name = col_name

    def sln_to_ecfp(self, X: pd.DataFrame, bit_vector: bool = True) -> pd.Series:
        """ Converts Sybil Line Notation to ECFP (Morgan algorithm)
        
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
            X[self.col_name].apply(lambda x: AllChem.MolFromSLN(x))
        if bit_vector == True:
            ss_ecfp = \
                ss_mols.apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, radius=2))
        else:
            ss_ecfp = \
                ss_mols.apply(lambda x: AllChem.GetMorganFingerprint(x, radius=2))
                
        return ss_ecfp


    def smiles_to_ecfp(self, X: pd.DataFrame, bit_vector: bool = True) -> pd.Series:
        """ Converts smiles to ECFP (Morgan algorithm)
        
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
            X[self.col_name].apply(lambda x: AllChem.MolFromSmiles(x))
        if bit_vector == True:
            ss_ecfp = \
                ss_mols.apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, radius=2))
        else:
            ss_ecfp = \
                ss_mols.apply(lambda x: AllChem.GetMorganFingerprint(x, radius=2))
                
        return ss_ecfp

    def sln_to_smiles(self, X: pd.DataFrame) -> pd.Series:
        """ Converts Sybil Line Notation to SMILES
        
        Parameters
        ----------
        X : pandas DataFrame
            Data with SMILES column

        Returns
        -------
        pandas Series:
            SLN transformed from SMILES
        """
        ss_smiles = X[self.col_name].values
        ss_smiles = ss_smiles.ravel()
        for i in range(len(ss_smiles)):
            ss_smiles[i] =  AllChem.MolFromSLN(ss_smiles[i])
            ss_smiles[i] = AllChem.MolToSmiles(ss_smiles[i])
            
        X[self.col_name] = ss_smiles                  
        return X[self.col_name]