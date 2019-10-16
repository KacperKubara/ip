"""EDA, ML pipeline script for Solubility Challenge Datasets"""
import pandas as pd 

from eda import DataDistribution
#from models import 
from data import load_sol_challenge

cols_to_analyse = [
    "Temperature", "assays", "Ionic Strength (M)",
    "S0 (mM)", "SD of S0 (mM)", "Kinetic Solubility (mM)",
    "SD of Kinetic Solubility (mM)"
    ]

if __name__ == "__main__":
    df = load_sol_challenge()
    print(df.head())
    # Data Preprocessing part - transform it into class later on
    df = df.dropna()
    print(len(df.index))
    # EDA
    #data_distribution = DataDistribution()