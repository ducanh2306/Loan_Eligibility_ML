"""
app.py
Streamlit web application for the Loan Eligibility Prediction model.
Run with:  streamlit run app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Make sure the project root is on sys.path ────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import DATA_PATH, MODEL_PATH, SCALER_PATH
from src.logger import get_logger
from src.data_loader import load_data
from src.preprocessing import impute_missing, encode_features, split_features_target
from src.train import run_training_pipeline, load_artifacts
from src.predict import predict_loan_eligibility

logger = get_logger(__name__)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="🏦",
    layout="wide",
)


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data
def get_raw_data():
    return load_data(DATA_PATH)


@st.cache_data
def get_processed_data():
    df = load_data(DATA_PATH)
    df = impute_missing(df)
    df = encode_features(df)
    X, y = split_features_target(df)
    return X, y


@st.cache_resource
def get_or_train_model(X, y):
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model, scaler = load_artifacts()
        return None, model, scaler, list(X.columns)
    results = run_training_pipeline(X, y)
    return results, results["best_model"], results["scaler"], results["feature_cols"]


# ── Sidebar nav ──────────────────────────────────────────────────────────────
st.sidebar.title("🏦 Loan Eligibility")
page = st.sidebar.radio(
    "Navigate",
    ["Data Explorer", "Model Training", "Predict"],
)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Data Explorer
# ════════════════════════════════════════════════════════════════════════════
if page == "Data Explorer":
    st.title("Data Explorer")
    st.markdown(
        "Explore the **credit.csv** used to build the loan prediction model."
        
    )

    df_raw = get_raw_data()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Applications", df_raw.shape[0])
    col2.metric("Features", df_raw.shape[1] - 1)
    approved = int((df_raw["Loan_Approved"] == "Y").sum())
    col3.metric("Approval Rate", f"{approved / len(df_raw):.1%}")

    st.subheader("Raw Data Sample")
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.subheader("Missing Values")
    missing = df_raw.isnull().sum().rename("Missing Count")
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values in the dataset.")
    else:
        st.bar_chart(missing)

    st.subheader("Target Distribution")
    fig, ax = plt.subplots(figsize=(4, 3))
    vc = df_raw["Loan_Approved"].value_counts()
    ax.bar(["Approved (Y)", "Denied (N)"], vc.values, color=["#2ecc71", "#e74c3c"])
    ax.set_ylabel("Count")
    ax.set_title("Loan Applications by Outcome")
    st.pyplot(fig, use_container_width=False)

    st.subheader("Numerical Feature Distributions")
    num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    fig2, axes = plt.subplots(1, len(num_cols), figsize=(14, 3))
    for ax, col in zip(axes, num_cols):
        df_raw[col].dropna().hist(ax=ax, bins=30, color="#3498db", edgecolor="white")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
    fig2.tight_layout()
    st.pyplot(fig2)

    st.subheader("Categorical Feature Counts")
    cat_cols = ["Gender", "Married", "Education", "Self_Employed",
                "Credit_History", "Property_Area"]
    fig3, axes = plt.subplots(2, 3, figsize=(14, 6))
    for ax, col in zip(axes.flat, cat_cols):
        vc2 = df_raw[col].value_counts()
        ax.bar(vc2.index.astype(str), vc2.values, color="#9b59b6")
        ax.set_title(col, fontsize=9)
        ax.tick_params(axis="x", labelsize=8, rotation=30)
    fig3.tight_layout()
    st.pyplot(fig3)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Training
# ════════════════════════════════════════════════════════════════════════════
elif page == "Model Training":
    st.title("🤖 Model Training & Evaluation")

    X, y = get_processed_data()

    artifacts_exist = os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)

    if artifacts_exist:
        st.info(
            "Pre-trained model found on disk. Click **Re-train** to rebuild from scratch.",
            icon="ℹ️",
        )

    run_btn = st.button("🚀 Train / Re-train Models", type="primary")

    if run_btn or not artifacts_exist:
        with st.spinner("Training three classifiers — this takes a few seconds …"):
            # Clear cache so we always re-run
            get_or_train_model.clear()
            results = run_training_pipeline(X, y)

        st.success("Training complete!", icon="✅")

        # Store results in session state so they persist without re-training
        st.session_state["results"] = results
        st.session_state["feature_cols"] = results["feature_cols"]

    # Display results (from session state or freshly computed)
    if "results" in st.session_state:
        results = st.session_state["results"]
        model_names = ["Logistic Regression", "Decision Tree", "Random Forest"]

        # ── Accuracy comparison ───────────────────────────────────────────
        st.subheader("Accuracy Comparison")
        acc_data = {
            name: {
                "Test Accuracy":   results[name]["eval"]["accuracy"],
                "CV Mean Accuracy": results[name]["cv"]["mean"],
            }
            for name in model_names
        }
        acc_df = pd.DataFrame(acc_data).T
        st.dataframe(acc_df.style.format("{:.4f}"), use_container_width=True)

        fig, ax = plt.subplots(figsize=(7, 3))
        x = np.arange(len(model_names))
        width = 0.35
        ax.bar(x - width/2, acc_df["Test Accuracy"],   width, label="Test", color="#2980b9")
        ax.bar(x + width/2, acc_df["CV Mean Accuracy"], width, label="CV Mean", color="#e67e22")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylim(0.6, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy Comparison")
        ax.legend()
        ax.axhline(0.76, color="red", linestyle="--", linewidth=0.9, label="Target (76%)")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

        # ── Per-model detail ──────────────────────────────────────────────
        st.subheader("Detailed Results per Model")
        tabs = st.tabs(model_names)
        for tab, name in zip(tabs, model_names):
            with tab:
                eval_res = results[name]["eval"]
                cv_res   = results[name]["cv"]

                c1, c2, c3 = st.columns(3)
                c1.metric("Test Accuracy", f"{eval_res['accuracy']:.4f}")
                c2.metric("CV Mean", f"{cv_res['mean']:.4f}")
                c3.metric("CV Std",  f"{cv_res['std']:.4f}")

                st.write("**CV Fold Scores:**",
                         ", ".join(f"{s:.4f}" for s in cv_res["scores"]))

                st.write("**Confusion Matrix:**")
                cm = eval_res["confusion_matrix"]
                fig_cm, ax_cm = plt.subplots(figsize=(3.5, 2.8))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Denied", "Approved"],
                    yticklabels=["Denied", "Approved"],
                    ax=ax_cm,
                )
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                ax_cm.set_title(f"{name} — Confusion Matrix")
                fig_cm.tight_layout()
                st.pyplot(fig_cm)

                st.write("**Classification Report:**")
                report_df = pd.DataFrame(eval_res["report"]).T
                st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

        # ── Feature importance (Random Forest) ────────────────────────────
        st.subheader("Feature Importances (Random Forest)")
        rf_model     = results["Random Forest"]["model"]
        feature_cols = results["feature_cols"]
        importances  = pd.Series(rf_model.feature_importances_, index=feature_cols)
        top_n        = importances.nlargest(15).sort_values()

        fig_fi, ax_fi = plt.subplots(figsize=(7, 5))
        top_n.plot(kind="barh", ax=ax_fi, color="#1abc9c")
        ax_fi.set_title("Top 15 Feature Importances")
        ax_fi.set_xlabel("Importance")
        fig_fi.tight_layout()
        st.pyplot(fig_fi)

    else:
        st.info("Click **Train / Re-train Models** to see results.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Predict
# ════════════════════════════════════════════════════════════════════════════
elif page == "Predict":
    st.title("Loan Eligibility Predictor")
    st.markdown(
        "Fill the form and see what the model predicts for a single applicant. "
    )

    # We need feature columns — try session state first, else recompute
    if "feature_cols" not in st.session_state:
        X, y = get_processed_data()
        _, _, _, feature_cols = get_or_train_model(X, y)
        st.session_state["feature_cols"] = feature_cols

    feature_cols = st.session_state["feature_cols"]

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender       = st.selectbox("Gender",        ["Male", "Female"])
            married      = st.selectbox("Married",       ["Yes", "No"])
            dependents   = st.selectbox("Dependents",    ["0", "1", "2", "3+"])
            education    = st.selectbox("Education",     ["Graduate", "Not Graduate"])
            self_emp     = st.selectbox("Self Employed", ["No", "Yes"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        with col2:
            applicant_income    = st.number_input("Applicant Income ($/mo)",   min_value=0,   value=5000, step=100)
            coapplicant_income  = st.number_input("Co-applicant Income ($/mo)", min_value=0,  value=0,    step=100)
            loan_amount         = st.number_input("Loan Amount (×$1000)",       min_value=1,  value=128,  step=1)
            loan_amount_term    = st.selectbox("Loan Term (months)",
                                               [360, 180, 480, 300, 240, 84, 120, 60, 36, 12])
            credit_history      = st.selectbox("Credit History", [1, 0],
                                               format_func=lambda x: "Has history (1)" if x == 1
                                                                      else "No history (0)")

        submitted = st.form_submit_button("Click here to predict", type="primary")

    if submitted:
        if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
            st.error(
                "No trained model found. Go to **🤖 Model Training** and train the model first.",
                icon="⚠️",
            )
        else:
            raw_input = {
                "Gender":             gender,
                "Married":            married,
                "Dependents":         dependents,
                "Education":          education,
                "Self_Employed":      self_emp,
                "ApplicantIncome":    applicant_income,
                "CoapplicantIncome":  float(coapplicant_income),
                "LoanAmount":         float(loan_amount),
                "Loan_Amount_Term":   float(loan_amount_term),
                "Credit_History":     float(credit_history),
                "Property_Area":      property_area,
            }

            try:
                result = predict_loan_eligibility(raw_input, feature_cols)
                prob   = result["probability"]
                label  = result["label"]

                if label == "Approved":
                    st.success(f"**{label}** — Congrats!! You got approval about: {prob:.1%}", icon="✅")
                else:
                    st.error(f"**{label}** — Opps!! You got denied about: {prob:.1%}", icon="❌")

                # Probability gauge
                fig_g, ax_g = plt.subplots(figsize=(5, 0.6))
                ax_g.barh(["P(Approved)"], [prob], color="#2ecc71" if label == "Approved" else "#e74c3c")
                ax_g.barh(["P(Approved)"], [1 - prob], left=[prob], color="#ecf0f1")
                ax_g.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
                ax_g.set_xlim(0, 1)
                ax_g.set_xlabel("Probability")
                fig_g.tight_layout()
                st.pyplot(fig_g)


            except Exception as exc:
                st.error(f"Prediction failed: {exc}", icon="⚠️")
                logger.exception("Prediction error")
