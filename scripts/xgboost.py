"""
Simplified XGBoost implementation

Prerequisites:
pandas >= 0.25.1
numpy >= 1.17.2
sklearn >= 0.22.1
xgboost >= 0.90
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer, confusion_matrix

# loading of dataset
datasetLocation = "../../dataset/TrafficLabelling/Friday-DDos-SHORTENED.csv"
currentDataframe = pd.read_csv(datasetLocation)

print(currentDataframe.shape)  # Verify rows/columns exist
print(currentDataframe.empty)  # True if no data
print(currentDataframe.head())