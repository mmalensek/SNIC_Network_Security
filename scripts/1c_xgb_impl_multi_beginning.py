"""
(1b/4)

Simplified XGBoost implementation - multiclass version (15 labels) with temporal split

Prerequisites:
xgboost >= 0.90
numpy >= 1.17.2
pandas >= 0.25.1
sklearn >= 0.22.1

Training:
- First half of Monday
- First half of Tuesday
- First half of Wednesday
- Thursday Morning
- Friday Morning

Testing:
- Second half of Monday
- Second half of Tuesday
- Second half of Wednesday
- Thursday Afternoon
- Friday Afternoon (DDoS + PortScan)
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from sklearn.preprocessing import LabelEncoder


# ============================================================
# LOAD + CLEAN CSV
# ============================================================

def load_and_clean_csv(path):

    print(f"\nLoading: {path}")

    # some CICIDS files are not UTF-8
    try:

        df = pd.read_csv(
            path,
            encoding="utf-8",
            low_memory=False
        )

    except UnicodeDecodeError:

        print("UTF-8 failed, trying latin1...")

        df = pd.read_csv(
            path,
            encoding="latin1",
            low_memory=False
        )

    # clean numeric columns
    for col in df.columns:

        if col != " Label":

            # convert to numeric
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            # clip huge values
            df[col] = df[col].clip(
                lower=-1e15,
                upper=1e15
            )

            # replace inf/nan
            df[col] = (
                df[col]
                .replace([np.inf, -np.inf], 0)
                .fillna(0)
            )

    print(f"Loaded {len(df)} rows")

    return df


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def temporal_split(df, split_ratio=0.5):

    split_idx = int(len(df) * split_ratio)

    train_df = df.iloc[:split_idx].copy()

    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df


# ============================================================
# MAIN
# ============================================================

def main():

    dataset_dir = "../../dataset/TrafficLabelling"

    # ========================================================
    # LOAD WHOLE-DAY FILES
    # ========================================================

    monday = load_and_clean_csv(
        os.path.join(
            dataset_dir,
            "Monday-WorkingHours.pcap_ISCX.csv"
        )
    )

    tuesday = load_and_clean_csv(
        os.path.join(
            dataset_dir,
            "Tuesday-WorkingHours.pcap_ISCX.csv"
        )
    )

    wednesday = load_and_clean_csv(
        os.path.join(
            dataset_dir,
            "Wednesday-workingHours.pcap_ISCX.csv"
        )
    )

    # ========================================================
    # TEMPORAL SPLITS
    # ========================================================

    monday_train, monday_test = temporal_split(monday)

    tuesday_train, tuesday_test = temporal_split(tuesday)

    wednesday_train, wednesday_test = temporal_split(wednesday)

    # ========================================================
    # LOAD MORNING FILES (TRAIN)
    # ========================================================

    thursday_morning = load_and_clean_csv(
        os.path.join(
            dataset_dir,
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
        )
    )

    friday_morning = load_and_clean_csv(
        os.path.join(
            dataset_dir,
            "Friday-WorkingHours-Morning.pcap_ISCX.csv"
        )
    )

    # ========================================================
    # LOAD AFTERNOON FILES (TEST)
    # ========================================================

    thursday_afternoon = load_and_clean_csv(
        os.path.join(
            dataset_dir,
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
        )
    )

    friday_afternoon_ddos = load_and_clean_csv(
        os.path.join(
            dataset_dir,
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        )
    )

    friday_afternoon_portscan = load_and_clean_csv(
        os.path.join(
            dataset_dir,
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
        )
    )

    # ========================================================
    # COMBINE TRAIN DATA
    # ========================================================

    train_df = pd.concat([
        monday_train,
        tuesday_train,
        wednesday_train,
        thursday_morning,
        friday_morning
    ], ignore_index=True)

    # ========================================================
    # COMBINE TEST DATA
    # ========================================================

    test_df = pd.concat([
        monday_test,
        tuesday_test,
        wednesday_test,
        thursday_afternoon,
        friday_afternoon_ddos,
        friday_afternoon_portscan
    ], ignore_index=True)

    # ========================================================
    # DATASET INFO
    # ========================================================

    print("\n================================================")
    print("DATASET INFORMATION")
    print("================================================")

    print(f"\nTraining rows: {len(train_df)}")

    print(f"Testing rows: {len(test_df)}")

    print("\nTraining labels:")
    print(train_df[" Label"].value_counts())

    print("\nTesting labels:")
    print(test_df[" Label"].value_counts())

    # ========================================================
    # HANDLE UNSEEN TEST CLASSES
    # ========================================================

    train_classes = set(
        train_df[" Label"].unique()
    )

    test_classes = set(
        test_df[" Label"].unique()
    )

    unseen_classes = test_classes - train_classes

    print("\n================================================")
    print("UNSEEN TEST CLASSES")
    print("================================================")

    if len(unseen_classes) == 0:

        print("\nNo unseen classes found.")

        filtered_test_df = test_df.copy()

    else:

        print("\nRemoving unseen classes:")

        for c in unseen_classes:
            print(c)

        # remove unseen classes from test set
        filtered_test_df = test_df[
            ~test_df[" Label"].isin(unseen_classes)
        ].copy()

    print(f"\nFiltered test rows: {len(filtered_test_df)}")

    # ========================================================
    # FEATURES
    # ========================================================

    X_train = train_df.drop(
        " Label",
        axis=1
    ).copy()

    X_test = filtered_test_df.drop(
        " Label",
        axis=1
    ).copy()

    # remove constant columns
    valid_columns = X_train.columns[
        X_train.nunique() > 1
    ]

    X_train = X_train[valid_columns]

    X_test = X_test[valid_columns]

    print(f"\nNumber of features: {X_train.shape[1]}")

    # ========================================================
    # LABEL ENCODING
    # ========================================================

    label_encoder = LabelEncoder()

    # IMPORTANT:
    # fit ONLY on training labels
    label_encoder.fit(
        train_df[" Label"]
    )

    y_train = label_encoder.transform(
        train_df[" Label"]
    )

    y_test = label_encoder.transform(
        filtered_test_df[" Label"]
    )

    class_names = label_encoder.classes_

    num_classes = len(class_names)

    print("\n================================================")
    print("CLASS INFORMATION")
    print("================================================")

    print(f"\nNumber of classes: {num_classes}")

    print("\nClass mapping:")

    for idx, label in enumerate(class_names):

        print(f"{idx}: {label}")

    # ========================================================
    # MODEL
    # ========================================================

    print("\n================================================")
    print("TRAINING MODEL")
    print("================================================")

    model = xgb.XGBClassifier(

        objective="multi:softprob",

        num_class=num_classes,

        n_estimators=100,

        max_depth=6,

        learning_rate=0.1,

        min_child_weight=5,

        subsample=0.8,

        colsample_bytree=0.8,

        eval_metric="mlogloss",

        early_stopping_rounds=10,

        random_state=42,

        tree_method="hist",

        n_jobs=-1
    )

    model.fit(

        X_train,
        y_train,

        eval_set=[
            (X_test, y_test)
        ],

        verbose=True
    )

    # ========================================================
    # BEST MODEL INFO
    # ========================================================

    print("\n================================================")
    print("BEST MODEL")
    print("================================================")

    print(f"\nBest iteration: {model.best_iteration}")

    print(f"Best score: {model.best_score}")

    # ========================================================
    # PREDICTIONS
    # ========================================================

    print("\n================================================")
    print("RUNNING PREDICTIONS")
    print("================================================")

    y_pred = model.predict(X_test)

    y_proba = model.predict_proba(X_test)

    # ========================================================
    # EVALUATION
    # ========================================================

    print("\n================================================")
    print("EVALUATION")
    print("================================================")

    bal_acc = balanced_accuracy_score(
        y_test,
        y_pred
    )

    print(f"\nBalanced Accuracy: {bal_acc:.6f}")

    # ========================================================
    # ROC AUC
    # ========================================================

    try:

        auc_score = roc_auc_score(

            y_test,

            y_proba,

            multi_class="ovo",

            average="macro"
        )

        print(f"ROC AUC (OVO Macro): {auc_score:.6f}")

    except Exception as e:

        print(f"\nROC AUC failed: {e}")

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n================================================")
    print("CLASSIFICATION REPORT")
    print("================================================")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            digits=4
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print("\n================================================")
    print("CONFUSION MATRIX")
    print("================================================")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    print("\n================================================")
    print("SAVING MODEL")
    print("================================================")

    os.makedirs(
        "classifier",
        exist_ok=True
    )

    model_path = (
        "classifier/xgb_temporal_model.json"
    )

    model.save_model(model_path)

    label_mapping_path = (
        "classifier/label_mapping.txt"
    )

    with open(
        label_mapping_path,
        "w",
        encoding="utf-8"
    ) as f:

        for idx, label in enumerate(class_names):

            f.write(f"{idx}: {label}\n")

    print(f"\nSaved model to: {model_path}")

    print(
        f"Saved label mapping to: "
        f"{label_mapping_path}"
    )

    print("\nFinished.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()