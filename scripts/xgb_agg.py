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
import os
import numpy as np
import pandas as pd
import xgboost as xgb

modelLocation = "classifier/xgb_model.json"
datasetLocation = "../../dataset/TrafficLabelling/Traffic-COMBINED.csv"
json_log_dir = "json_log"

np.set_printoptions(suppress=True, precision=6)

def main():

    print("\nPrediction selection by label..")
    printSettings = int(input("\nPrint every row separately (1), print json (2), print both (3): "))
    print("")

    # load the trained classifier
    model = xgb.XGBClassifier()
    model.load_model(modelLocation)
    print("Model loaded..")

    # loading the dataset
    dataframe = pd.read_csv(datasetLocation)
    print("Dataset loaded..")

    # show label distribution and ask set selection for test
    label_counts = dataframe[" Label"].value_counts()
    print("\nAvailable labels and counts:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")

    selected_input = input("\nEnter labels to include in test (comma-separated), or 'all': ").strip()
    if selected_input.lower() == "all" or selected_input == "":
        selected_labels = label_counts.index.tolist()
    else:
        selected_labels = [l.strip() for l in selected_input.split(",") if l.strip()]

    invalid_labels = [l for l in selected_labels if l not in label_counts.index]
    if invalid_labels:
        raise ValueError(f"Invalid label(s) provided: {invalid_labels}. Choose from {list(label_counts.index)}")

    filtered_df = dataframe[dataframe[" Label"].isin(selected_labels)].reset_index(drop=True)
    if filtered_df.empty:
        raise ValueError("No rows match selected label(s).")

    dataframe = filtered_df

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

    # set a true label array for accuracy testing
    original_labels = dataframe[" Label"].copy()
    true_labels = original_labels.copy()

    # compute actual majority for the true label
    majority_label = true_labels.value_counts().idxmax()
    majority_ratio = float(true_labels.value_counts().max() / len(true_labels))
    print("True label calculated..")
    
    # row selection for prediction (all selected rows)
    test_rows = X
    true_labels = y

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
            "idle_mean": float(test_rows["Idle Mean"].mean()),
            "active_mean": float(test_rows["Active Mean"].mean())
        }
    }

    output = {
        "selected_labels": selected_labels,
        "model_prediction": final_prediction,
        "confidence": confidence,
        "avg_attack_probability": round(avg_attack_prob, 4),
        "features": aggregated_features
    }

    ground_truth_output = {
        "most_common_true_label": majority_label,
        "true_label_ratio": majority_ratio
    }

    # save to json_log folder with timestamped filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"json_log/prediction_{timestamp}.json"    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"JSON saved to: {filename}")

    # save ground truth to separate json file
    ground_truth_filename = f"json_log/ground_truth_{timestamp}.json"
    with open(ground_truth_filename, 'w') as f:
        json.dump(ground_truth_output, f, indent=2)
    print(f"Ground truth JSON saved to: {ground_truth_filename}")

    # printing of row data / json based on settings
    if(printSettings == 2 or printSettings == 3):
        print("\nFinal JSON output:")
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
