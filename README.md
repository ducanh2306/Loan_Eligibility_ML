# Loan Eligibility Prediction

**CST2216 Individual Term Project — Modularizing and Deploying ML Code**

A modular, production-style machine learning project that predicts whether a loan application should be approved, built from the Jupyter Notebook solution and deployed as a Streamlit web application. https://loaneligibilityan.streamlit.app/

---

## Project Structure

```
loan_eligibility/
├── app.py                  # Streamlit web application (entry point)
├── requirements.txt
├── README.md
├── data/
│   └── credit.csv          # German Credit dataset (614 rows × 13 columns)
├── logs/
│   └── app.log             # Auto-generated runtime log
├── src/
│   ├── __init__.py
│   ├── config.py           # Centralised configuration (paths, params)
│   ├── logger.py           # Logging setup (file + console)
│   ├── data_loader.py      # CSV loading & validation
│   ├── preprocessing.py    # Imputation, encoding, scaling
│   ├── train.py            # Model training, evaluation, persistence
│   └── predict.py          # Single-applicant inference
└── tests/
    └── test_pipeline.py    # Pytest unit tests
```

---

## Quick Start

### 1. Clone / open the project

```bash
cd loan_eligibility
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` with three pages:

| Page | What it does |
|------|-------------|
| 📊 Data Explorer | Dataset overview, distributions, missing values |
| 🤖 Model Training | Train LR / Decision Tree / Random Forest, compare accuracy, view confusion matrices and feature importances |
| 🔮 Predict | Enter applicant details and get an instant approval prediction |

### 5. Run tests

```bash
python -m pytest tests/ -v
```

---

## Models

| Model | Test Accuracy | CV Mean |
|-------|-------------|---------|
| Logistic Regression | ~81% | ~80% |
| Decision Tree | ~74% | ~75% |
| **Random Forest** | **~80%** | **~80%** |

The **Random Forest** model is saved as the default predictor (`data/model.pkl`).

---

## Dataset

German Credit / Loan Eligibility dataset — 614 applicants, 13 columns.

**Target:** `Loan_Approved` (Y / N)

**Key features:** Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area

---

## Dependencies

- Python ≥ 3.10
- streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn, pytest

