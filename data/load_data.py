""" Utility functions to load the data"""
import pandas as pd 

path = "/home/kacper/Desktop/Github/ip/data/solubility_challenge_train.csv"

def load_sol_challenge() -> pd.DataFrame:
    """ Load Solubility Challenge dataset"""
    return pd.read_csv(path, delimiter=",")