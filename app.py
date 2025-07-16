import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load saved components
model = joblib.load("model.pkl")
explainer = joblib.load("explainer.pkl")
X_test = pd.read_csv("X_test.csv")
shap_values = np.load("shap_values.npy")

st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

st.title("🏦 Loan Approval Predictor with Explainability")
st.write("🔍 Understand why a loan was approved or rejected using SHAP values.")

# Pick applicant
index = st.selectbox("Select Applicant ID", options=range(len(X_test)), format_func=lambda x: f"Applicant #{x}")

# Predict
sample = X_test.iloc[[index]]
prediction = model.predict(sample)[0]
label = "✅ Approved" if prediction == 1 else "❌ Rejected"

st.subheader(f"Prediction: {label}")

# Plot SHAP explanation
st.subheader("🔎 Feature Impact (SHAP Bar Plot)")
shap_values_row = shap_values[index, :, 1]

explanation = shap.Explanation(
    values=shap_values_row,
    base_values=explainer.expected_value[1],
    data=sample.iloc[0],
    feature_names=X_test.columns.tolist()
)

fig, ax = plt.subplots()
shap.plots.bar(explanation, show=False)
st.pyplot(fig)
