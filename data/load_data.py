""" Utility functions to load the data"""
import pandas as pd 

from config import PATH_SOL_DATA, PATH_WANG_DATA

def load_sol_challenge() -> pd.DataFrame:
    """ Loads Solubility Challenge dataset"""
    return pd.read_csv(PATH_SOL_DATA)

def load_wang_data() -> pd.DataFrame:
    """ Loads Wang dataset"""
    return pd.read_csv(PATH_WANG_DATA)