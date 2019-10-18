""" EDA, ML pipeline script for Solubility Challenge Datasets"""
import pandas as pd 

from eda import DataDistribution, FeatureCorrelation
from data import load_sol_challenge
from preprocessing import PreProcessor

cols_to_analyse = [
    "Temperature", "assays", "Ionic Strength (M)",
    "S0 (mM)", "SD of S0 (mM)", "Kinetic Solubility (mM)",
    "SD of Kinetic Solubility (mM)"
    ]
cols_with_str = [
    "Ionic Strength (M)", "SD of S0 (mM)", "Kinetic Solubility (mM)",
    "SD of Kinetic Solubility (mM)"   
]

if __name__ == "__main__":
    df = load_sol_challenge()

    # Data Preprocessing
    preprocessor = PreProcessor()
    df = preprocessor.str_to_float(df, cols_with_str)
    df = preprocessor.remove_nans(df)

    # EDA
    # Data Distribution
    data_distribution = DataDistribution(cols_to_analyse, ignore_outliers=False)
    data_distribution.run(df)
    # Feature Correlation
    feature_correlation = FeatureCorrelation(cols_to_analyse, figsize=(9, 9))
    feature_correlation.run(df)