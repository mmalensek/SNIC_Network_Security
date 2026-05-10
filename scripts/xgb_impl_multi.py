"""
(1b/4)

Simplified XGBoost implementation - multiclass version (15 labels)

Prerequisites:
xgboost >= 0.90
numpy >= 1.17.2
pandas >= 0.25.1
sklearn >= 0.22.1
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
from sklearn.model_selection import train_test_split


def main():
    # loading of dataset
    datasetLocation = "../../dataset/TrafficLabelling/Traffic-COMBINED.csv"
    dataframe = pd.read_csv(datasetLocation)

    # converting all string columns except label to numeric
    for col in dataframe.columns:
        if col != " Label":
            dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")
            dataframe[col] = dataframe[col].clip(lower=-1e15, upper=1e15)
            dataframe[col] = dataframe[col].replace([np.inf, -np.inf], 0).fillna(0)

    # loading input columns to X and target variable to y
    X = dataframe.drop(" Label", axis=1).copy()
    X = X.loc[:, X.nunique() > 1]

    # encode ALL labels, not just BENIGN vs rest
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(dataframe[" Label"])

    class_names = label_encoder.classes_
    num_classes = len(class_names)

    print("")
    print(f"Number of classes: {num_classes}")
    print("\nLabel mapping:")
    for idx, label in enumerate(class_names):
        print(f"{idx}: {label}")

    print("\nLabel distribution (%):")
    print(pd.Series(y).value_counts(normalize=True).sort_index())

    print("\nFeature correlations with label (top 10 highest):")
    corr_series = X.corrwith(pd.Series(y), method="spearman").abs()
    high_corr_features = corr_series.nlargest(10)
    for feature, corr in high_corr_features.items():
        print(f"  {feature}: {corr:.4f}")

    print("")

    # splitting training data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42,
        stratify=y,
        test_size=0.25
    )

    classification_xgb = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=50,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        seed=42,
        early_stopping_rounds=5,
        eval_metric="mlogloss"
    )

    classification_xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )

    print(f"Best iteration: {classification_xgb.best_iteration}")
    print(f"Best validation score (mlogloss): {classification_xgb.best_score:.6f}")

    # predictions
    y_pred = classification_xgb.predict(X_test)
    y_proba = classification_xgb.predict_proba(X_test)

    # evaluation
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"\nBalanced accuracy: {bal_acc:.6f}")

    try:
        auc_ovo = roc_auc_score(y_test, y_proba, multi_class="ovo", average="macro")
        print(f"Multiclass ROC AUC (OVO, macro): {auc_ovo:.6f}")
    except Exception as e:
        print(f"ROC AUC could not be computed: {e}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # save label mapping
    os.makedirs("classifier", exist_ok=True)

    label_mapping_path = "classifier/label_mapping.txt"
    with open(label_mapping_path, "w", encoding="utf-8") as f:
        for idx, label in enumerate(class_names):
            f.write(f"{idx}: {label}\n")

    # save the model
    model_path = "classifier/xgb_model_multiclass.json"
    classification_xgb.save_model(model_path)

    print(f"\nSaved the model to {model_path}")
    print(f"Saved label mapping to {label_mapping_path}")


if __name__ == "__main__":
    main()
