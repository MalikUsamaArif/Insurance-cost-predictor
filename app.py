import sys
import sklearn.compose

# Patch the missing class so scikit-learn 1.8.0 can unpickle the 1.6.1 model file
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
    sys.modules['sklearn.compose._column_transformer']._RemainderColsList = _RemainderColsList

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("best_insurance_model.joblib")

model = load_model()

# ── CDC BMI helper (must match training) ──────────────────────────────────────
def cdc_bmi_category(bmi):
    if bmi < 18.5:   return "Underweight"
    elif bmi < 25.0: return "Normal"
    elif bmi < 30.0: return "Overweight"
    elif bmi < 35.0: return "Obese_Class_I"
    elif bmi < 40.0: return "Obese_Class_II"
    else:            return "Obese_Class_III"

def age_group(age):
    if age <= 25:   return "18-25"
    elif age <= 35: return "26-35"
    elif age <= 45: return "36-45"
    elif age <= 55: return "46-55"
    else:           return "55+"

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("**Assignment-02 | CLO-2 — Powered by Gradient Boosting + sklearn Pipeline**")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Patient Information")
    age      = st.slider("Age",               18, 64, 35)
    bmi      = st.slider("BMI",               15.0, 55.0, 27.0, step=0.1)
    children = st.slider("Number of Children", 0, 5, 0)
    sex      = st.selectbox("Sex",    ["male", "female"])
    smoker   = st.selectbox("Smoker", ["no", "yes"])
    region   = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

with col2:
    st.subheader("Real-Time Prediction")

    bmi_cat      = cdc_bmi_category(bmi)
    age_grp      = age_group(age)
    smk_obese    = int(smoker == "yes" and bmi >= 30)
    high_risk_fl = int(smoker == "yes" or bmi >= 35)
    fam_size     = "none" if children == 0 else ("small" if children <= 2 else "large")

    patient = pd.DataFrame([{
        "age": age, "bmi": bmi, "children": children,
        "sex": sex, "smoker": smoker, "region": region,
        "bmi_category": bmi_cat, "age_group": age_grp,
        "smoker_obese": smk_obese, "high_risk": high_risk_fl,
        "family_size": fam_size
    }])

    prediction = model.predict(patient)[0]

    st.metric("Predicted Annual Insurance Cost", f"${prediction:,.2f}")
    st.metric("Monthly Equivalent",             f"${prediction/12:,.2f}")

    st.divider()
    st.markdown("**Patient Risk Summary**")
    st.write(f"- CDC BMI Category : **{bmi_cat}**")
    st.write(f"- Age Group        : **{age_grp}**")
    st.write(f"- High Risk Flag   : **{'Yes' if high_risk_fl else 'No'}**")
    st.write(f"- Smoker + Obese   : **{'Yes' if smk_obese else 'No'}**")

    if smoker == "yes":
        st.warning("Smoking significantly increases insurance costs — typically 3-4x higher.")
    if bmi >= 30:
        st.info(f"BMI {bmi:.1f} falls in the **{bmi_cat}** category (CDC classification).")

st.divider()
st.caption("Model: sklearn Pipeline | Features: age, bmi, children, sex, smoker, region + engineered features")
