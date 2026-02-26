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

    print("\nPrediction row selection..")
    leftBoundary = int(input("Enter lower bound: "))
    rightBoundary = int(input("Enter higher boundary: "))
    printSettings = int(input("\nPrint every row separately (1), print json (2), print both (3): "))
    print("")

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
    test_rows = X.iloc[leftBoundary:rightBoundary]
    true_labels = y[leftBoundary:rightBoundary]

    # prediction per flow
    predictions = model.predict(test_rows)
    probabilities = model.predict_proba(test_rows)

    print("\n------------------------------------\n")

    if(printSettings == 1 or printSettings == 3):
        # printing of results per flow
        for i in range(len(test_rows)):
            print(f"Row {i}")
            print("True label:", true_labels[i])
            print("Predicted:", predictions[i])
            print("Probabilities [BENIGN, ATTACK]:", probabilities[i])
            print("")
        print("------------------------------------")

    # getting data for json output
    avg_attack_prob = float(np.mean(probabilities[:, 1]))
    final_prediction = "ATTACK" if avg_attack_prob > 0.5 else "BENIGN"
    confidence = avg_attack_prob if final_prediction == "ATTACK" else 1 - avg_attack_prob
    syn_count = float(test_rows[" SYN Flag Count"].sum())
    ack_count = float(test_rows[" ACK Flag Count"].sum())
    syn_ack_ratio = syn_count / ack_count if ack_count != 0 else syn_count
    total_fwd = float(test_rows[" Total Fwd Packets"].sum())
    total_bwd = float(test_rows[" Total Backward Packets"].sum())
    fwd_bwd_ratio = total_fwd / total_bwd if total_bwd != 0 else total_fwd
   
    # aggregated prediction, formatting for json
    aggregated_features = {
        "traffic_summary": {
            "total_flows": int(len(test_rows)),
            "avg_flow_duration": float(test_rows[" Flow Duration"].mean()),
            "packets_per_second": float(test_rows[" Flow Packets/s"].mean()),
            "bytes_per_second": float(test_rows["Flow Bytes/s"].mean()),
            "total_fwd_packets": total_fwd,
            "total_bwd_packets": total_bwd,
            "fwd_bwd_ratio": round(fwd_bwd_ratio, 3)
        },

        "tcp_flags": {
            "syn_count": syn_count,
            "ack_count": ack_count,
            "rst_count": float(test_rows[" RST Flag Count"].sum()),
            "fin_count": float(test_rows["FIN Flag Count"].sum()),
            "syn_ack_ratio": round(syn_ack_ratio, 3)
        },

        "packet_statistics": {
            "avg_packet_size": float(test_rows[" Average Packet Size"].mean()),
            "packet_length_std": float(test_rows[" Packet Length Std"].mean()),
            "min_packet_length": float(test_rows[" Min Packet Length"].mean()),
            "max_packet_length": float(test_rows[" Max Packet Length"].mean())
        },

        "timing": {
            "flow_iat_mean": float(test_rows[" Flow IAT Mean"].mean()),
            "flow_iat_std": float(test_rows[" Flow IAT Std"].mean()),
            "idle_mean": float(test_rows[" Idle Mean"].mean()),
            "active_mean": float(test_rows[" Active Mean"].mean())
        }
    }

    output = {
        "window_left": leftBoundary,
        "window_right": rightBoundary,
        "model_prediction": final_prediction,
        "confidence": confidence,
        "avg_attack_probability": round(avg_attack_prob, 4),
        "features": aggregated_features
    }

    # printing of row data / json based on settings
    if(printSettings == 2 or printSettings == 3):
        print("\nFinal JSON output:")
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()