"""EDA, ML pipeline script for Solubility Challenge Datasets"""
import pandas as pd 

from eda import DataDistribution
#from models import 
from data import load_sol_challenge

if __name__ == "__main__":
    df = load_sol_challenge()
    print(df.head())
    # Data Preprocessing part - transform it into class later on
    
    # EDA
    data_distribution = DataDistribution()