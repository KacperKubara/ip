""" EDA, ML pipeline script for Wang dataset"""
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, train_test_split 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor

from eda import DataDistribution, FeatureCorrelation
from data import load_wang_data
from preprocessing import PreProcessor
from config import PATH_RESULTS_WANG_DATA, PATH_RESULTS_EDA_CORR_WANG_DATA,\
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
X_names_num = [
    "Expt",
    "Polarizability"    
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
    # Decrease dataset size - just for testing purposes
    df = df[:2000]
    print(f"DF size: {len(df.index)}")
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
        rand_search = RandomizedSearchCV(model, params, n_iter=50, n_jobs=-1,
                                        cv=10, scoring=scorer, random_state=42)
        # Random Search fit
        rand_search.fit(X_train, y_train)
        # Use best estimator for predicitons on validation data
        y_pred = rand_search.predict(X_valid)
        # Save results data
        results[model_name] = {}
        results[model_name]["MAE"] = {mae(y_valid, y_pred)}
        results[model_name]["Var_of_y_pred"] = {np.var(y_pred)}
        print(f"Model: {model_name}")
        print(results[model_name])

    pd.DataFrame(results).to_csv(PATH_RESULTS_WANG_DATA + "/results.csv")
    print(f"Averaged MAE Result: {results}")    