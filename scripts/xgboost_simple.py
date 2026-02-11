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
    datasetLocation = "../../dataset/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" # temp dataset
    dataframe = pd.read_csv(datasetLocation)

    # converting all string to numeric
    for col in dataframe.columns:
        if col != " Label":  # Skip target
            # encountering an error string becomes NaN
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
            dataframe[col] = dataframe[col].clip(lower=-1e15, upper=1e15)  # Prevent inf
            dataframe[col] = dataframe[col].replace([np.inf, -np.inf], 0).fillna(0)

    # loading input columns to X and target variable to y
    X = dataframe.drop(" Label", axis = 1).copy()
    le = LabelEncoder()
    y = le.fit_transform(dataframe[" Label"])

    # splitting training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    classification_xgb = xgb.XGBClassifier(objective="binary:logistic", seed=42, early_stopping_rounds=10, eval_metric="aucpr", base_score=0.5)
    classification_xgb.fit(X_train, y_train, verbose=True, eval_set=[(X_test, y_test)])

if __name__ == "__main__":
    main()