""" EDA, ML pipeline script for Solubility Challenge Datasets"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold 
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from xgboost import XGBRegressor

from eda import DataDistribution, FeatureCorrelation
from data import load_sol_challenge
from preprocessing import PreProcessor
from utils import average

cols_to_analyse = [
    "Temperature", "assays", "Ionic Strength (M)",
    "S0 (mM)", "SD of S0 (mM)", "Kinetic Solubility (mM)",
    "SD of Kinetic Solubility (mM)"
    ]
cols_with_str = [
    "Ionic Strength (M)", "SD of S0 (mM)",
    "Kinetic Solubility (mM)","SD of Kinetic Solubility (mM)"   
]
y_name = "Kinetic Solubility (mM)"
X_names = [
    'Substance', 'Temperature', 'assays',
    'Ionic Strength (M)', 'S0 (mM)', 'SD of S0 (mM)',
    'SMILES', 'InChI'
       ]
X_names_num = [
    'Temperature', 'assays', 'Ionic Strength (M)',
    'S0 (mM)', 'SD of S0 (mM)'
        ]
models = {
    "lasso": {
        "model": Lasso(),
        "params": []
    },
    "SVR": {
        "model": SVR(),
        "params": []
    },
    "RandomForest": {
        "model": RandomForestRegressor(),
        "params": []
    },
    "XGBoost": {
        "model": XGBRegressor(),
        "params": []
    }
}
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

    # Get independent and dependent variables
    X = np.asarray(df[X_names_num])
    y = np.asarray(df[y_name])
    # Small amount of data -> use KFold CV
    kfold = KFold(n_splits = 5, random_state=42)
    # Define models
    lasso = Lasso()
    results = []
    for train_ix, test_ix in kfold.split(X):
        # Fit
        lasso.fit(X[train_ix], y[train_ix])
        # Predict
        y_pred = lasso.predict(X[test_ix])
        print(f"y_pred: {y_pred}")
        print(f"y_test: {y[test_ix]}")
        # Compute MSE
        result_temp = mse(y[test_ix], y_pred)
        print(f"Result_temp: {result_temp}")
        results.append(result_temp)
    results_average = average(results)
    print(f"Averaged MSE Result: {results_average}")