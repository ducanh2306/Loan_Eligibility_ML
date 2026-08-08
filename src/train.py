"""
Model training, cross-validation, evaluation, and persistence.
"""

import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.config import (
    TEST_SIZE, RANDOM_STATE, RF_PARAMS, CV_FOLDS,
    MODEL_PATH, SCALER_PATH,
)
from src.preprocessing import fit_scaler, transform_scaler
from src.logger import get_logger

logger = get_logger(__name__)


# ── Train / test split ───────────────────────────────────────────────────────

def split_data(X: pd.DataFrame, y: pd.Series):
    """Stratified 80/20 train-test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    logger.info(
        "Split: train=%d  test=%d  (stratified, test_size=%.0f%%)",
        len(X_train), len(X_test), TEST_SIZE * 100,
    )
    return X_train, X_test, y_train, y_test


# ── Individual model trainers ────────────────────────────────────────────────

def train_logistic_regression(X_train_scaled, y_train):
    logger.info("Training Logistic Regression …")
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model


def train_decision_tree(X_train_scaled, y_train):
    logger.info("Training Decision Tree …")
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    return model


def train_random_forest(X_train_scaled, y_train):
    logger.info("Training Random Forest …")
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train_scaled, y_train)
    return model


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model, X_test_scaled, y_test, model_name: str = "Model") -> dict:
    y_pred = model.predict(X_test_scaled)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info("%s  →  Accuracy: %.4f", model_name, acc)
    logger.info("Confusion matrix:\n%s", cm)
    return {"accuracy": acc, "confusion_matrix": cm, "report": report}


def cross_validate_model(model, X_train_scaled, y_train,
                         model_name: str = "Model") -> dict:
    kfold  = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold)
    logger.info(
        "%s  CV-%d  mean=%.4f  std=%.4f",
        model_name, CV_FOLDS, scores.mean(), scores.std(),
    )
    return {"scores": scores, "mean": scores.mean(), "std": scores.std()}


# ── Persistence ──────────────────────────────────────────────────────────────

def save_artifacts(model, scaler):
    """Pickle the best model and scaler to disk."""
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved model → %s", MODEL_PATH)
    logger.info("Saved scaler → %s", SCALER_PATH)


def load_artifacts():
    """Load the persisted model and scaler."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Loaded model and scaler from disk.")
    return model, scaler


# ── Full training pipeline (called by Streamlit or CLI) ──────────────────────

def run_training_pipeline(X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_test, y_train, y_test = split_data(X, y)
    scaler, X_train_scaled = fit_scaler(X_train)
    X_test_scaled          = transform_scaler(scaler, X_test)

    results = {}

    # Logistic Regression
    lr = train_logistic_regression(X_train_scaled, y_train)
    results["Logistic Regression"] = {
        "model":  lr,
        "eval":   evaluate_model(lr,  X_test_scaled, y_test, "Logistic Regression"),
        "cv":     cross_validate_model(lr, X_train_scaled, y_train, "Logistic Regression"),
    }

    # Decision Tree
    dt = train_decision_tree(X_train_scaled, y_train)
    results["Decision Tree"] = {
        "model": dt,
        "eval":  evaluate_model(dt, X_test_scaled, y_test, "Decision Tree"),
        "cv":    cross_validate_model(dt, X_train_scaled, y_train, "Decision Tree"),
    }

    # Random Forest
    rf = train_random_forest(X_train_scaled, y_train)
    results["Random Forest"] = {
        "model": rf,
        "eval":  evaluate_model(rf, X_test_scaled, y_test, "Random Forest"),
        "cv":    cross_validate_model(rf, X_train_scaled, y_train, "Random Forest"),
    }

    # Persist best model (Random Forest)
    save_artifacts(rf, scaler)

    results["best_model"]    = rf
    results["scaler"]        = scaler
    results["feature_cols"]  = list(X_train.columns)

    return results
