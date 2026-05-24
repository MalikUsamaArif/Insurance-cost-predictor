# 🏥 Medical Insurance Cost Predictor
### Assignment-02 | CLO-2 — Multivariate Regression | Production-Grade ML

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-NAME.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Pipeline-orange?logo=scikit-learn)
![MLflow](https://img.shields.io/badge/MLflow-Tracked-blue?logo=mlflow)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-red)
![License](https://img.shields.io/badge/License-MIT-green)

> **Live Demo →** [Click here to try the app](https://YOUR-APP-NAME.streamlit.app)  
> Predict your annual medical insurance cost in real-time using a trained Gradient Boosting model.

---

## 📌 Problem Statement

Medical insurance pricing is complex and depends on many patient-level factors. This project builds a **multivariate regression system** that predicts individual annual insurance charges using demographic and health features — combining production-grade ML engineering with domain-aware feature design.

---

## ✨ Key Features & Enhancements

| # | Enhancement | Description |
|---|---|---|
| 1 | **sklearn Pipeline** | Full `ColumnTransformer` → model pipeline. Zero data leakage, exportable `.joblib` |
| 2 | **SHAP Interpretability** | Summary plot, bar plot, waterfall plot — explains *why* each patient is charged more |
| 3 | **Streamlit Web App** | Live interactive app with sliders, dropdowns, and real-time prediction |
| 4 | **MLflow Tracking** | All 6 model runs logged — hyperparameters, R², RMSE, artifacts |
| 5 | **CDC BMI Classification** | Official 6-tier CDC obesity scale + business impact metrics |

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Medical Cost Personal Dataset (Kaggle / UCI) |
| Records | 1,338 rows |
| Features | 7 raw + 6 engineered |
| Target | `charges` — Annual insurance cost (USD) |

### Feature Mix (CLO-2 Requirement)

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Age of primary beneficiary |
| `bmi` | Numerical | Body Mass Index |
| `children` | Ordinal | Number of dependants (0–5) |
| `sex` | Categorical | Gender (male / female) |
| `smoker` | Categorical | Smoking status (yes / no) |
| `region` | Categorical | US residential region (4 areas) |
| `bmi_category` | Ordinal (CDC) | Underweight → Obese Class III |
| `age_group` | Ordinal | 18-25 → 55+ |
| `smoker_obese` | Binary flag | Smoker AND BMI ≥ 30 |
| `high_risk` | Binary flag | Smoker OR BMI ≥ 35 |
| `family_size` | Categorical | none / small / large |

---

## 🏆 Model Results

| Model | CV R² | Test R² | Test MAE | Test RMSE |
|---|---|---|---|---|
| **Gradient Boosting** ⭐ | 0.9776 | **0.9821** | $1,509 | $1,962 |
| XGBoost | 0.9765 | 0.9813 | $1,561 | $2,005 |
| Random Forest | 0.9780 | 0.9805 | $1,585 | $2,044 |
| Linear Regression | 0.9783 | 0.9798 | $1,663 | $2,083 |
| Lasso Regression | 0.9782 | 0.9797 | $1,668 | $2,088 |
| Ridge Regression | 0.9745 | 0.9778 | $1,768 | $2,181 |

**Best Model: Gradient Boosting — R² = 0.9821 (explains 98.2% of cost variance)**

---

## 💼 Business Impact

| Metric | Baseline (Mean Price) | Our Model |
|---|---|---|
| MAE per customer | ~$11,000+ | ~$1,509 |
| At 100,000 customers | — | **~$950M** in reduced pricing errors/year |

---

## 🔍 SHAP Model Interpretability

SHAP (SHapley Additive exPlanations) explains every individual prediction:

- `smoker` is the #1 driver — pushes predictions up by $15,000–$23,000
- `bmi` combined with smoking creates a non-linear cost spike
- `age` has a steady positive effect across all patients

---

## 📁 Project Structure

```
insurance-cost-predictor/
│
├── app.py                          # Streamlit web application
├── best_insurance_model.joblib     # Trained pipeline (preprocessor + model)
├── insurance.csv                   # Dataset
├── Insurance_Enhanced.ipynb        # Full notebook (10 steps)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR-USERNAME/insurance-cost-predictor.git
cd insurance-cost-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```

### 4. View MLflow experiment dashboard
```bash
mlflow ui
# Open http://localhost:5000
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
xgboost
shap
mlflow
streamlit
joblib
matplotlib
seaborn
```

---

## 🧠 Pipeline Architecture

```
Raw Patient Data
      │
      ▼
ColumnTransformer
  ├── Numerical (age, bmi, children)    → StandardScaler
  ├── Binary (sex, smoker)              → OrdinalEncoder
  ├── Nominal (region, family_size)     → OneHotEncoder
  ├── CDC BMI (bmi_category)            → OrdinalEncoder (risk order)
  └── Age Group (age_group)             → OrdinalEncoder
      │
      ▼
  Gradient Boosting Regressor
      │
      ▼
  Predicted Insurance Charge (USD)
```

---

## 👤 Author

**Your Name**  
Department of Computer Science  
Assignment-02 | CLO-2 | Multivariate Regression

---

## 📄 License

This project is licensed under the MIT License.
