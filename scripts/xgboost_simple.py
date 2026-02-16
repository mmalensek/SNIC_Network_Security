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
    datasetLocation = "../../dataset/TrafficLabelling/Traffic-WorkingHours-COMBINED.csv"
    dataframe = pd.read_csv(datasetLocation)

    # converting all string to numeric
    for col in dataframe.columns:
        if col != " Label":  # skip the target column
            # encountering an error string transforms it to NaN
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
            dataframe[col] = dataframe[col].clip(lower=-1e15, upper=1e15)
            dataframe[col] = dataframe[col].replace([np.inf, -np.inf], 0).fillna(0)

    # loading input columns to X and target variable to y
    X = dataframe.drop(" Label", axis = 1).copy()
    X = X.loc[:, X.nunique() > 1]
    y = np.where(dataframe[" Label"] == "BENIGN", 0, 1) # encodes BENIGN as 0 and anything else as 1

    # diagnosis
    print("")
    print("Label distribution (%):")
    print(pd.Series(y).value_counts(normalize=True).sort_index())
    print("\nFeature correlations with label (top 10 highest):")
    corr_series = X.corrwith(pd.Series(y), method='spearman').abs()
    high_corr_features = corr_series.nlargest(10)
    for feature, corr in high_corr_features.items():
        print(f"  {feature}: {corr:.4f}")

    print("")

    # splitting training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    classification_xgb = xgb.XGBClassifier(
        objective="binary:logistic", 
        n_estimators=50,         
        max_depth=3,             
        min_child_weight=5,       
        subsample=0.8,            
        colsample_bytree=0.8,     
        seed=42, 
        early_stopping_rounds=5,   
        eval_metric="aucpr"
    )

    classification_xgb.fit(
        X_train, y_train, 
        eval_set=[(X_test, y_test)],
        verbose=True
    )
    print(f"Best iteration: {classification_xgb.best_iteration}")
    print(f"Final AUCPR: {classification_xgb.best_score:.6f}")


if __name__ == "__main__":
    main()