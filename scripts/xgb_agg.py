"""
XGBoost classifier aggregator (FIXED feature alignment)
"""

import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb

modelLocation = "classifier/xgb_model.json"
featureNamesLocation = "classifier/feature_names.json"
datasetLocation = "../../dataset/TrafficLabelling/Traffic-COMBINED.csv"
json_log_dir = "json_log"

np.set_printoptions(suppress=True, precision=6)


def main():

    print("\nPrediction selection by label..")
    printSettings = int(input("\nPrint every row separately (1), print json (2), print both (3): "))
    print("")

    # load model
    model = xgb.XGBClassifier()
    model.load_model(modelLocation)
    print("Model loaded..")

    # load feature names (CRUCIAL FIX)
    with open(featureNamesLocation, "r") as f:
        model_features = json.load(f)
    print("Feature names loaded..")

    # load dataset
    dataframe = pd.read_csv(datasetLocation)
    print("Dataset loaded..")

    # show label distribution
    label_counts = dataframe[" Label"].value_counts()
    print("\nAvailable labels and counts:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")

    # user selection
    selected_input = input("\nEnter labels to include in test (comma-separated), or 'all': ").strip()

    if selected_input.lower() == "all" or selected_input == "":
        selected_labels = label_counts.index.tolist()
    else:
        selected_labels = [l.strip() for l in selected_input.split(",") if l.strip()]

    invalid_labels = [l for l in selected_labels if l not in label_counts.index]
    if invalid_labels:
        raise ValueError(f"Invalid label(s): {invalid_labels}")

    dataframe = dataframe[dataframe[" Label"].isin(selected_labels)].reset_index(drop=True)

    if dataframe.empty:
        raise ValueError("No rows match selected label(s).")

    # preprocessing
    for col in dataframe.columns:
        if col != " Label":
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
            dataframe[col] = dataframe[col].clip(lower=-1e15, upper=1e15)
            dataframe[col] = dataframe[col].replace([np.inf, -np.inf], 0).fillna(0)

    print("Dataset preprocessed..")

    # split
    X = dataframe.drop(" Label", axis=1).copy()
    y = np.where(dataframe[" Label"] == "BENIGN", 0, 1)

    # ✅ FIXED FEATURE ALIGNMENT
    print("\nAligning features...")

    # add missing
    missing = [c for c in model_features if c not in X.columns]
    if missing:
        print(f"Adding missing columns: {missing}")
        for c in missing:
            X[c] = 0

    # drop extra
    extra = [c for c in X.columns if c not in model_features]
    if extra:
        print(f"Dropping extra columns: {extra}")
        X = X.drop(columns=extra)

    # reorder
    X = X[model_features]

    print(f"Final feature shape: {X.shape}")
    print("Feature alignment done.")

    # ground truth
    original_labels = dataframe[" Label"].copy()
    majority_label = original_labels.value_counts().idxmax()
    majority_ratio = float(original_labels.value_counts().max() / len(original_labels))

    # prediction
    test_rows = X
    true_labels = y

    predictions = model.predict(test_rows)
    probabilities = model.predict_proba(test_rows)

    print("\n------------------------------------\n")

    if printSettings in [1, 3]:
        for i in range(len(test_rows)):
            print(f"Row {i}")
            print("True label:", true_labels[i])
            print("Predicted:", predictions[i])
            print("Probabilities [BENIGN, ATTACK]:", probabilities[i])
            print("")
        print("------------------------------------")

    # aggregation
    avg_attack_prob = float(np.mean(probabilities[:, 1]))
    final_prediction = "ATTACK" if avg_attack_prob > 0.5 else "BENIGN"
    confidence = avg_attack_prob if final_prediction == "ATTACK" else 1 - avg_attack_prob

    syn_count = float(test_rows[" SYN Flag Count"].sum())
    ack_count = float(test_rows[" ACK Flag Count"].sum())
    syn_ack_ratio = syn_count / ack_count if ack_count != 0 else syn_count

    total_fwd = float(test_rows[" Total Fwd Packets"].sum())
    total_bwd = float(test_rows[" Total Backward Packets"].sum())
    fwd_bwd_ratio = total_fwd / total_bwd if total_bwd != 0 else total_fwd

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

    # ensure folder exists
    os.makedirs(json_log_dir, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{json_log_dir}/prediction_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    ground_truth_filename = f"{json_log_dir}/ground_truth_{timestamp}.json"
    with open(ground_truth_filename, 'w') as f:
        json.dump(ground_truth_output, f, indent=2)

    print(f"JSON saved to: {filename}")
    print(f"Ground truth JSON saved to: {ground_truth_filename}")

    if printSettings in [2, 3]:
        print("\nFinal JSON output:")
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()