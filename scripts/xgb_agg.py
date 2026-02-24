"""
XGBoost classifier aggregator

Prerequisites:
xgboost >= 0.90
numpy >= 1.17.2
pandas >= 0.25.1
sklearn >= 0.22.1
"""

import numpy as np
import pandas as pd
import xgboost as xgb

modelLocation = "classifier/xgb_model.json"
datasetLocation = "../../dataset/TrafficLabelling/Traffic-COMBINED.csv"

def main():

    # load the trained classifier
    model = xgb.XGBClassifier()
    model.load_model(modelLocation)
    print("Model loaded..")

    # loading the dataset
    dataframe = pd.read_csv(datasetLocation)
    print("Dataset loaded..")

    # apply same preprocessing as in training
    for col in dataframe.columns:
        if col != " Label":
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
            dataframe[col] = dataframe[col].clip(lower=-1e15, upper=1e15)
            dataframe[col] = dataframe[col].replace([np.inf, -np.inf], 0).fillna(0)

    # prepare X and y exactly like in training
    X = dataframe.drop(" Label", axis=1).copy()
    X = X.loc[:, X.nunique() > 1]
    y = np.where(dataframe[" Label"] == "BENIGN", 0, 1)
    print("Dataset preprocessed..")


    # row selection for prediction
    print("Prediction row selection..")
    leftBoundary = input("Enter lower bound: ")
    rightBoundary = input("Enter higher boundary: ")
    test_rows = X.iloc[leftBoundary:rightBoundary]
    true_labels = y[leftBoundary:rightBoundary]

    # prediction
    predictions = model.predict(test_rows)
    probabilities = model.predict_proba(test_rows)

    # printing of results
    for i in range(len(test_rows)):
        print(f"\nRow {i}")
        print("True label:", true_labels[i])
        print("Predicted:", predictions[i])
        print("Probabilities [BENIGN, ATTACK]:", probabilities[i])


if __name__ == "__main__":
    main()