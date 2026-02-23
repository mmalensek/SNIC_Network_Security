"""
XGBoost classifier aggregator

Prerequisites:
xgboost >= 0.90
numpy >= 1.17.2
pandas >= 0.25.1
sklearn >= 0.22.1
"""

import xgboost as xgb

def main():

    # load the trained classifier
    model = xgb.XGBClassifier()
    model.load_model("classifier/xgb_model.json")
    print("Upam da dela brez problemov :)")

if __name__ == "__main__":
    main()