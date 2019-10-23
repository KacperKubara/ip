""" Utility functions to load the data"""
import pandas as pd 

from config import PATH_SOL_DATA

def load_sol_challenge() -> pd.DataFrame:
    """ Load Solubility Challenge dataset"""
    return pd.read_csv(PATH_SOL_DATA, delimiter=",")