"""
(2/4)

XGBoost classifier aggregator (label-based selection + classifier selection)

Prerequisites:
xgboost >= 0.90
numpy >= 1.17.2
pandas >= 0.25.1
sklearn >= 0.22.1
json
"""

import json
import os
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb

datasetLocation = "../../dataset/TrafficLabelling/Traffic-COMBINED.csv"
JSON_LOG_DIR = "json_log/1_groundtruth_and_xgboost_prediction"

np.set_printoptions(suppress=True, precision=6)

AVAILABLE_MODELS = {
    "binary": {
        "path": "classifier/xgb_model.json",
        "type": "binary",
        "description": "Binary classifier (BENIGN vs ATTACK)"
    },
    "multiclass": {
        "path": "classifier/xgb_model_multiclass.json",
        "type": "multiclass",
        "description": "15-class classifier"
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="XGBoost classifier aggregator with selectable classifier"
    )
    parser.add_argument(
        "--model-key",
        type=str,
        choices=list(AVAILABLE_MODELS.keys()),
        help="Predefined model key to load"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Custom path to a model file (.json). Overrides --model-key path if provided"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["binary", "multiclass"],
        help="Model type for custom --model-path"
    )
    return parser.parse_args()


def choose_model(args):
    # Case 1: custom path passed manually
    if args.model_path:
        if not args.model_type:
            raise ValueError("When using --model-path, you must also provide --model-type (binary or multiclass).")
        return {
            "key": "custom",
            "path": args.model_path,
            "type": args.model_type,
            "description": "Custom user-provided model"
        }

    # Case 2: predefined key passed in CLI
    if args.model_key:
        selected = AVAILABLE_MODELS[args.model_key].copy()
        selected["key"] = args.model_key
        return selected

    # Case 3: interactive selection
    print("\nAvailable classifiers:")
    model_keys = list(AVAILABLE_MODELS.keys())
    for i, key in enumerate(model_keys):
        info = AVAILABLE_MODELS[key]
        print(f"{i}: {key} -> {info['description']} ({info['path']})")

    selected_index = int(input("\nSelect classifier index: ").strip())
    selected_key = model_keys[selected_index]

    selected = AVAILABLE_MODELS[selected_key].copy()
    selected["key"] = selected_key
    return selected


def generate_outputs(model, model_type, test_rows, true_labels, true_label_names, selected_labels, printSettings, suffix, dataframe):
    print(f"\nGenerating output set {suffix}...")

    predictions = model.predict(test_rows)
    probabilities = model.predict_proba(test_rows)

    print("\n------------------------------------\n")

    if printSettings == 1 or printSettings == 3:
        for i in range(len(test_rows)):
            print(f"Row {i}")
            print("True label:", true_labels.iloc[i])
            print("Predicted:", predictions[i])

            if model_type == "binary":
                print("Probabilities [BENIGN, ATTACK]:", probabilities[i])
            else:
                print("Probabilities [all classes]:", probabilities[i])

            print("")
        print("------------------------------------")

    # ==========================================
    # SINGLE SAMPLE MODE
    # ==========================================
    if len(test_rows) == 1:
        row_data = test_rows.iloc[0].to_dict()

        # convert numpy values to python native types
        row_data = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in row_data.items()
        }

        prediction = predictions[0]

        if model_type == "binary":
            probability_info = {
                "probabilities": {
                    "BENIGN": round(float(probabilities[0][0]), 4),
                    "ATTACK": round(float(probabilities[0][1]), 4)
                }
            }
        else:
            labels_sorted = sorted(dataframe[" Label"].unique())

            probability_info = {
                "probabilities": {
                    label: round(float(probabilities[0][idx]), 4)
                    for idx, label in enumerate(labels_sorted)
                }
            }

        output = {
            "classifier_used": model_type,
            "model_prediction": int(prediction) if isinstance(prediction, (np.integer, np.int64)) else prediction,
            "row_data": row_data
        }

        output.update(probability_info)

        ground_truth_output = {
            "true_label": true_label_names.iloc[0]
        }

        return output, ground_truth_output

    # ==========================================
    # MULTI SAMPLE MODE (original aggregation)
    # ==========================================

    majority_label = true_label_names.value_counts().idxmax()
    majority_ratio = float(true_label_names.value_counts().max() / len(true_label_names))

    if model_type == "binary":
        avg_attack_prob = float(np.mean(probabilities[:, 1]))
        final_prediction = "ATTACK" if avg_attack_prob > 0.5 else "BENIGN"
        confidence = avg_attack_prob if final_prediction == "ATTACK" else 1 - avg_attack_prob

        probability_summary = {
            "avg_attack_probability": round(avg_attack_prob, 4)
        }

    else:
        labels_sorted = sorted(dataframe[" Label"].unique())
        label_to_idx = {label: idx for idx, label in enumerate(labels_sorted)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        avg_class_probs = np.mean(probabilities, axis=0)
        final_class_idx = int(np.argmax(avg_class_probs))
        final_prediction = idx_to_label[final_class_idx]
        confidence = float(avg_class_probs[final_class_idx])

        probability_summary = {
            "avg_class_probabilities": [round(float(p), 4) for p in avg_class_probs],
            "predicted_class_index": final_class_idx,
            "predicted_class_label": final_prediction
        }

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
        "classifier_used": model_type,
        "confidence": round(float(confidence), 4),
        "features": aggregated_features
    }

    output.update(probability_summary)

    ground_truth_output = {
        "most_common_true_label": majority_label,
        "true_label_ratio": majority_ratio
    }

    return output, ground_truth_output


def main():
    args = parse_args()
    selected_model = choose_model(args)

    modelLocation = selected_model["path"]
    modelType = selected_model["type"]

    print("\nLoading model and dataset...")
    print(f"Selected classifier: {selected_model['key']}")
    print(f"Description: {selected_model['description']}")
    print(f"Model path: {modelLocation}")
    print(f"Model type: {modelType}")

    if not os.path.exists(modelLocation):
        raise FileNotFoundError(f"Model file not found: {modelLocation}")

    model = xgb.XGBClassifier()
    model.load_model(modelLocation)
    print("Model loaded...")

    dataframe = pd.read_csv(datasetLocation)
    print("Dataset loaded...")

    for col in dataframe.columns:
        if col != " Label":
            dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")
            dataframe[col] = dataframe[col].clip(lower=-1e15, upper=1e15)
            dataframe[col] = dataframe[col].replace([np.inf, -np.inf], 0).fillna(0)

    X = dataframe.drop(" Label", axis=1).copy()
    X = X.loc[:, X.nunique() > 1]
    original_labels = dataframe[" Label"].copy()

    # ground-truth encoding depends on selected classifier
    if modelType == "binary":
        y = np.where(dataframe[" Label"] == "BENIGN", 0, 1)
    else:
        labels_sorted = sorted(dataframe[" Label"].unique())
        label_to_idx = {label: idx for idx, label in enumerate(labels_sorted)}
        y = dataframe[" Label"].map(label_to_idx).astype(int).values

        print("\nMulticlass label mapping used in this script:")
        for label, idx in label_to_idx.items():
            print(f"{idx}: {label}")

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
            model_type=modelType,
            test_rows=test_rows,
            true_labels=true_labels,
            true_label_names=true_label_names,
            selected_labels=selected_labels,
            printSettings=printSettings,
            suffix=run_idx,
            dataframe=dataframe
        )

        filename = f"{JSON_LOG_DIR}/prediction_{timestamp}_{run_idx}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"JSON saved to: {filename}")

        ground_truth_filename = f"{JSON_LOG_DIR}/ground_truth_{timestamp}_{run_idx}.json"
        with open(ground_truth_filename, "w", encoding="utf-8") as f:
            json.dump(ground_truth_output, f, indent=2, ensure_ascii=False)
        print(f"Ground truth JSON saved to: {ground_truth_filename}")

        if printSettings == 2 or printSettings == 3:
            print(f"\nFinal JSON output {run_idx}:")
            print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
