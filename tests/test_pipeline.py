"""
tests/test_pipeline.py
Basic unit tests for the loan eligibility pipeline.
Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np

from src.data_loader import load_data
from src.preprocessing import impute_missing, encode_features, split_features_target, fit_scaler
from src.config import DATA_PATH, TARGET_COL


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    return load_data(DATA_PATH)


@pytest.fixture(scope="module")
def processed_df(raw_df):
    df = impute_missing(raw_df)
    df = encode_features(df)
    return df


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDataLoader:
    def test_load_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_expected_rows(self, raw_df):
        assert raw_df.shape[0] == 614

    def test_target_column_exists(self, raw_df):
        assert TARGET_COL in raw_df.columns

    def test_target_values(self, raw_df):
        assert set(raw_df[TARGET_COL].unique()).issubset({"Y", "N"})


class TestPreprocessing:
    def test_no_missing_after_impute(self, raw_df):
        df = impute_missing(raw_df)
        assert df.isnull().sum().sum() == 0

    def test_encode_drops_loan_id(self, raw_df):
        df = impute_missing(raw_df)
        df = encode_features(df)
        assert "Loan_ID" not in df.columns

    def test_target_binary_after_encode(self, processed_df):
        assert set(processed_df[TARGET_COL].unique()).issubset({0, 1})

    def test_feature_split_shapes(self, processed_df):
        X, y = split_features_target(processed_df)
        assert X.shape[0] == y.shape[0]
        assert TARGET_COL not in X.columns

    def test_scaler_output_range(self, processed_df):
        X, y = split_features_target(processed_df)
        scaler, X_scaled = fit_scaler(X)
        assert X_scaled.min() >= -1e-9   # allow floating-point slack
        assert X_scaled.max() <= 1 + 1e-9
