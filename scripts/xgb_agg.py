"""
XGBoost classifier aggregator

Prerequisites:
xgboost >= 0.90
numpy >= 1.17.2
pandas >= 0.25.1
sklearn >= 0.22.1
json
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb

modelLocation = "classifier/xgb_model.json"
datasetLocation = "../../dataset/TrafficLabelling/Traffic-COMBINED.csv"

np.set_printoptions(suppress=True, precision=6)

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
    leftBoundary = int(input("Enter lower bound: "))
    rightBoundary = int(input("Enter higher boundary: "))
    test_rows = X.iloc[leftBoundary:rightBoundary]
    true_labels = y[leftBoundary:rightBoundary]

    # prediction per flow
    predictions = model.predict(test_rows)
    probabilities = model.predict_proba(test_rows)

    # printing of results per flow
    print("------------------------------------")
    for i in range(len(test_rows)):
        print(f"\nRow {i}")
        print("True label:", true_labels[i])
        print("Predicted:", predictions[i])
        print("Probabilities [BENIGN, ATTACK]:", probabilities[i])
    print("------------------------------------")

    # aggregated prediction
    avg_attack_prob = float(np.mean(probabilities[:, 1]))
    final_prediction = "ATTACK" if avg_attack_prob > 0.5 else "BENIGN"
    confidence = avg_attack_prob if final_prediction == "ATTACK" else 1 - avg_attack_prob
    aggregated_features = {
        "total_flows": int(len(test_rows)),
        "flow duration": float(test_rows[" Flow Duration"].mean()),
        "total fwd packets": float(test_rows[" Total Fwd Packets"].mean()),
        "fwd packet length": float(test_rows["Total Length of Fwd Packets"].mean()),
        "flow bytes per second": float(test_rows["Flow Bytes/s"].mean()),
    }

    # json output of aggregated predictions
    output = {
        "window_left": leftBoundary,
        "window_right": rightBoundary,
        "model_prediction": final_prediction,
        "confidence": round(float(confidence), 4),
        "features": aggregated_features
    }

    print("\nFinal JSON output:")
    print(json.dumps(output, indent=2))




if __name__ == "__main__":
    main()