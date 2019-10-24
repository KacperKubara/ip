import pandas as pd
import numpy as np

from eda import DataDistribution, FeatureCorrelation
from data import load_wang_data
from preprocessing import PreProcessor
from config import PATH_RESULTS, PATH_RESULTS_EDA_CORR_WANG_DATA,\
    PATH_RESULTS_EDA_DIST_WANG_DATA

cols_to_analyse = [
    "Expt", "ClogP",
    "Polarizability"
    ]
y_name = "ClogP"
X_str = [
    "Name",
    "SLN"
    ]
X_names = [
    "Expt",
    "Polarizability"    
]

if __name__ == "__main__":
    df = load_wang_data()
    # Data Preprocessing
    preprocessor = PreProcessor()
    df = preprocessor.remove_nans(df)
    
    # EDA
    # Data Distribution
    data_distribution = DataDistribution(cols_to_analyse, PATH_RESULTS_EDA_DIST_WANG_DATA, 
                                        ignore_outliers=False)
    data_distribution.run(df)
    # Feature Correlation
    feature_correlation = FeatureCorrelation(cols_to_analyse, PATH_RESULTS_EDA_CORR_WANG_DATA, 
                                            figsize=(9, 9))
    feature_correlation.run(df)