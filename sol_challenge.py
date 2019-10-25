""" EDA, ML pipeline script for Solubility Challenge Datasets"""
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold 
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from eda import DataDistribution, FeatureCorrelation
from data import load_sol_challenge
from preprocessing import PreProcessor
from config import PATH_RESULTS, PATH_RESULTS_EDA_CORR,\
    PATH_RESULTS_EDA_DIST

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

models_dict = {
    "lasso": {
        "model": Lasso(),
        "params": {
            "alpha": np.random.uniform(low=0.01, high=3, size=25),
            "normalize": [True],
            "random_state": [42]
        }
    },
    "SVR": {
        "model": SVR(gamma='auto'),
        "params": {
            "kernel": ['rbf', 'poly'],
            "degree": [2,3,4,5],
            "gamma": np.logspace(1e-7, 1e-1, 25),
            "C": np.logspace(1e-2, 10e4, 25),
            "epsilon": np.logspace(1e-2, 1, 25)
        }
    },
    "RandomForest": {
        "model": RandomForestRegressor(),
        "params": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 10, 15, 20, 25, 30],
            "min_samples_split": [2, 5, 10, 15, 100],
            "min_samples_leaf": [1, 2, 5, 10] 
        }
    },
    "XGBoost": {
        "model": XGBRegressor(),
        "params": {
            "gamma": np.random.uniform(low=0.01, high=0.05, size=10),
            "max_depth": [4, 5, 6],
            "min_child_weight": [4, 5, 6],
            "reg_alpha": [1e-5, 1e-2, 0.1, 1, 10, 100]
        }
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
    data_distribution = DataDistribution(cols_to_analyse, PATH_RESULTS_EDA_DIST, ignore_outliers=False)
    data_distribution.run(df)
    # Feature Correlation
    feature_correlation = FeatureCorrelation(cols_to_analyse, PATH_RESULTS_EDA_CORR, figsize=(9, 9))
    feature_correlation.run(df)

    # Get independent and dependent variables
    X = np.asarray(df[X_names_num])
    y = np.asarray(df[y_name])
    # Get validation and training set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2) 

    # Iterate over models
    results = {}
    scorer = make_scorer(mae, False)
    for model_name in models_dict.keys():
        model = models_dict[model_name]["model"]
        params = models_dict[model_name]["params"]
        rand_search = RandomizedSearchCV(model, params, n_iter=10, n_jobs=-1,
                                        cv=10, scoring=scorer, random_state=42)
        # Random Search fit
        rand_search.fit(X_train, y_train)
        # Use best estimator for predicitons on validation data
        y_pred = rand_search.predict(X_valid)
        # Save results data
        results[model_name] = {}
        results[model_name]["MAE"] = {mae(y_valid, y_pred)}
        results[model_name]["Val_set_variance"] = {np.var(y_valid)}
        print(f"Model: {model_name}")
        print(results[model_name])

    pd.DataFrame(results).to_csv(PATH_RESULTS + "/results.csv")
    print(f"Averaged MAE Result: {results}")    