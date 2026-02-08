"""
Simplified XGBoost implementation

Prerequisites:
pandas >= 0.25.1
numpy >= 1.17.2
sklearn >= 0.22.1
xgboost >= 0.90
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer, confusion_matrix

def main():

    # loading of dataset
    datasetLocation = "../../dataset/TrafficLabelling/Friday-DDos-SHORTENED.csv" # temp dataset
    dataframe = pd.read_csv(datasetLocation)

    # loading input columns to X and target variable to y
    X = dataframe.drop(" Label", axis = 1).copy()
    y = dataframe[" Label"].copy()

    # splitting training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    classification_xgb = xgb.XGBClassifier(objective="binary:logistic", seed=42)
    classification_xgb.fit(X_train, y_train, verbose=True, early_stopping_rounds=10, eval_metric="aucpr", eval_set=[(X_test, y_test)])

if __name__ == "__main__":
    main()