"""
Simplified XGBoost implementation

Prerequisites:
xgboost >= 0.90
numpy >= 1.17.2
pandas >= 0.25.1
sklearn >= 0.22.1
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV


def main():

    # loading of dataset
    datasetLocation = "../../dataset/TrafficLabelling/Friday-DDos-SHORTENED.csv" # temp dataset
    dataframe = pd.read_csv(datasetLocation)

    # loading input columns to X and target variable to y
    X = dataframe.drop(" Label", axis = 1).copy()
    le = LabelEncoder()
    y = le.fit_transform(dataframe[" Label"])

    # splitting training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    classification_xgb = xgb.XGBClassifier(objective="binary:logistic", seed=42, early_stopping_rounds=10, eval_metric="aucpr")
    classification_xgb.fit(X_train, y_train, verbose=True, eval_set=[(X_test, y_test)])

if __name__ == "__main__":
    main()