"""Preprocessor Class"""
import pandas as pd 
import numpy as np 

class PreProcessor():
    """ Preprocesses the dataset"""

    def __init__(self):
        pass

    def remove_nans(self, X: pd.DataFrame) -> pd.DataFrame:
        """ Remove rows with NaN values
        
        Parameters
        ----------
        X : pandas DataFrame
            Data which has to be preprocessed

        Returns
        -------
        pd.DataFrame : 
            Data without NaN values
        """
        return X.dropna()

    def str_to_float(self, X: pd.DataFrame, col_names: list) -> pd.DataFrame:
        """ Converts string number representation to floats.
        If not possible, replace with NaN
        
        Parameters
        ----------
        X : pandas DataFrame
            Data which has to be preprocessed

        col_names: [str]
            List of column names that should be preprocessed

        Returns
        -------
        pd.DataFrame : 
            Data with elements of float type without
            unwanted string values
        """
        for col in col_names:
            X[col] = X[col].map(self._lambda_func_str_to_float)
        return X

    def _lambda_func_str_to_float(self, val):
        """ CHelper function for str_to_float()
        
        Parameters
        ----------
        val : float, int, or str
            Data element that should be converted
            into float value

        Returns
        -------
        float, np.nan:
            if conversion is successful, it returns float.
            Otherwise, it will return NaN
        """
        try:
            return float(val)
        except ValueError:
            return np.nan