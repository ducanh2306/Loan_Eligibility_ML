"""
This file contains all configuration settings for project
--> Easy to fix if anything changes in the future
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "credit.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "data")
LOG_DIR    = os.path.join(BASE_DIR, "logs")

# ── Model persistence ────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# ── Data settings ────────────────────────────────────────────────────────────
TARGET_COL   = "Loan_Approved"
DROP_COLS    = ["Loan_ID"]
TEST_SIZE    = 0.20
RANDOM_STATE = 42

CATEGORICAL_COLS = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area",
]
TREAT_AS_CATEGORICAL = ["Credit_History", "Loan_Amount_Term"]

# ── Model settings ───────────────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
}

CV_FOLDS = 5

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE  = os.path.join(LOG_DIR, "app.log")
LOG_LEVEL = "INFO"
