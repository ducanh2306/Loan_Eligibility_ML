"""
preprocessing.py
All data cleaning, imputation, encoding, and scaling logic --> Validated data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    TARGET_COL,
    DROP_COLS,
    CATEGORICAL_COLS,
    TREAT_AS_CATEGORICAL,
)
from src.logger import get_logger

logger = get_logger(__name__)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
    - Categorical / treat-as-categorical -> mode
    - LoanAmount (numerical with outliers) -> median

    Handles both string and numeric storage of columns like
    Dependents, Credit_History, and Loan_Amount_Term.
    """
    df = df.copy()
    missing_before = df.isnull().sum().sum()
    logger.info("Missing values before imputation: %d", missing_before)

    # Categorical columns -> mode (CoW-safe assignment)
    for col in CATEGORICAL_COLS + TREAT_AS_CATEGORICAL:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.debug("Imputed '%s' with mode: %s", col, mode_val)

    # LoanAmount -> median (robust to outliers)
    if "LoanAmount" in df.columns and df["LoanAmount"].isnull().any():
        med = df["LoanAmount"].median()
        df["LoanAmount"] = df["LoanAmount"].fillna(med)
        logger.debug("Imputed 'LoanAmount' with median: %.2f", med)

    missing_after = df.isnull().sum().sum()
    logger.info("Missing values after imputation: %d", missing_after)
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical features and binary-encode the target.
    """
    df = df.copy()

    # Dependents may be stored as float (0.0, 1.0, 2.0, 3.0) in the CSV.
    # Convert to clean string labels so get_dummies produces readable column names.
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].apply(
            lambda x: str(int(x)) if pd.notna(x) else x
        )

    # Convert treat-as-categorical columns to object so get_dummies encodes them
    for col in TREAT_AS_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype("object")

    # Drop ID column(s)
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
            logger.debug("Dropped column: %s", col)

    # One-hot encode categorical features (exclude target)
    ohe_cols = [c for c in CATEGORICAL_COLS + TREAT_AS_CATEGORICAL if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols)
    logger.info("Shape after one-hot encoding: %s", df.shape)

    # Binary-encode target: Y -> 1, N -> 0
    df[TARGET_COL] = df[TARGET_COL].map({"Y": 1, "N": 0})

    return df


def split_features_target(df: pd.DataFrame):
    """
    Separate feature matrix X and target vector y.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    logger.info("Features: %d  |  Target: %s", X.shape[1], TARGET_COL)
    return X, y


def fit_scaler(X_train: pd.DataFrame):
    """
    Fit a MinMaxScaler on training data and transform it.

    Returns:
        (fitted scaler, scaled training array)
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)
    logger.info("Scaler fitted on %d training samples.", X_train.shape[0])
    return scaler, X_scaled


def transform_scaler(scaler: MinMaxScaler, X: pd.DataFrame) -> np.ndarray:
    """
    Apply a pre-fitted scaler to new data (test set or single prediction).
    """
    return scaler.transform(X)


def preprocess_single_input(raw_dict: dict, feature_columns: list,
                             scaler: MinMaxScaler) -> np.ndarray:
    
    df_input = pd.DataFrame([raw_dict])

    # Normalise Dependents to string
    if "Dependents" in df_input.columns:
        df_input["Dependents"] = df_input["Dependents"].apply(
            lambda x: str(int(x)) if pd.notna(x) else x
        )

    # Cast treat-as-categorical
    for col in TREAT_AS_CATEGORICAL:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype("object")

    # One-hot encode
    ohe_cols = [c for c in CATEGORICAL_COLS + TREAT_AS_CATEGORICAL
                if c in df_input.columns]
    df_input = pd.get_dummies(df_input, columns=ohe_cols)

    # Align columns to training feature set (fills missing dummies with 0)
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)

    scaled = scaler.transform(df_input)
    return scaled
