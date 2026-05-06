"""
(2/4)

XGBoost classifier aggregator (label-based selection)

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
JSON_LOG_DIR = "json_log/1_groundtruth_and_xgboost_prediction"

np.set_printoptions(suppress=True, precision=6)


def generate_outputs(model, test_rows, true_labels, true_label_names, selected_labels, printSettings, suffix):
    print(f"\nGenerating output set {suffix}...")

    majority_label = true_label_names.value_counts().idxmax()
    majority_ratio = float(true_label_names.value_counts().max() / len(true_label_names))

    predictions = model.predict(test_rows)
    probabilities = model.predict_proba(test_rows)

    print("\n------------------------------------\n")

    if printSettings == 1 or printSettings == 3:
        for i in range(len(test_rows)):
            print(f"Row {i}")
            print("True label:", true_labels.iloc[i])
            print("Predicted:", predictions[i])
            print("Probabilities [BENIGN, ATTACK]:", probabilities[i])
            print("")
        print("------------------------------------")

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

    return output, ground_truth_output


def main():
    print("\nLoading model and dataset...")

    model = xgb.XGBClassifier()
    model.load_model(modelLocation)
    print("Model loaded...")

    dataframe = pd.read_csv(datasetLocation)
    print("Dataset loaded...")

    for col in dataframe.columns:
        if col != " Label":
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
            dataframe[col] = dataframe[col].clip(lower=-1e15, upper=1e15)
            dataframe[col] = dataframe[col].replace([np.inf, -np.inf], 0).fillna(0)

    X = dataframe.drop(" Label", axis=1).copy()
    X = X.loc[:, X.nunique() > 1]
    y = np.where(dataframe[" Label"] == "BENIGN", 0, 1)
    original_labels = dataframe[" Label"].copy()

    print("Dataset preprocessed...")

    print("\nAvailable labels in dataset:")
    labels = dataframe[" Label"].unique()
    for i, label in enumerate(labels):
        print(f"{i}: {label}")

    selected_indices = input("\nSelect labels to include (e.g. 0,2,3): ")
    selected_indices = [int(i.strip()) for i in selected_indices.split(",")]
    selected_labels = [labels[i] for i in selected_indices]

    print("\nSelected labels:", selected_labels)

    mask = dataframe[" Label"].isin(selected_labels)
    filtered_rows = X[mask]
    filtered_true_labels = pd.Series(y, index=dataframe.index)[mask]
    filtered_true_label_names = original_labels[mask]

    limit = int(input("\nHow many samples to test per output (0 = all): "))
    n = int(input("How many different prediction/groundtruth pairs to generate: "))
    printSettings = int(input("\nPrint every row separately (1), print json (2), print both (3): "))
    print("")

    if len(filtered_rows) == 0:
        print("No rows found for selected labels.")
        return

    os.makedirs(JSON_LOG_DIR, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    for run_idx in range(1, n + 1):
        if limit > 0:
            sample_size = min(limit, len(filtered_rows))
            sampled_indices = np.random.choice(
                filtered_rows.index,
                size=sample_size,
                replace=False
            )
            test_rows = filtered_rows.loc[sampled_indices]
            true_labels = filtered_true_labels.loc[sampled_indices]
            true_label_names = filtered_true_label_names.loc[sampled_indices]
        else:
            test_rows = filtered_rows.copy()
            true_labels = filtered_true_labels.copy()
            true_label_names = filtered_true_label_names.copy()

        output, ground_truth_output = generate_outputs(
            model=model,
            test_rows=test_rows,
            true_labels=true_labels,
            true_label_names=true_label_names,
            selected_labels=selected_labels,
            printSettings=printSettings,
            suffix=run_idx
        )

        filename = f"{JSON_LOG_DIR}/prediction_{timestamp}_{run_idx}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"JSON saved to: {filename}")

        ground_truth_filename = f"{JSON_LOG_DIR}/ground_truth_{timestamp}_{run_idx}.json"
        with open(ground_truth_filename, 'w') as f:
            json.dump(ground_truth_output, f, indent=2)
        print(f"Ground truth JSON saved to: {ground_truth_filename}")

        if printSettings == 2 or printSettings == 3:
            print(f"\nFinal JSON output {run_idx}:")
            print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
