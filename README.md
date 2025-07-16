# 🏦 Loan Approval Predictor with SHAP Explainability

An interactive machine learning app built using **Streamlit** to predict loan approval decisions and explain the reasoning behind them using **SHAP** (SHapley Additive exPlanations).

---

## 📌 Project Overview

This project demonstrates how machine learning can be applied in the **fintech domain** to automate loan approval while maintaining **transparency and interpretability**.

Users can:
- Predict whether a loan will be approved or rejected
- Understand which features influenced the decision using SHAP bar plots

---

## 📊 Features

- ✅ Trained with **Random Forest Classifier**
- ⚙️ **Hyperparameter tuning** using `GridSearchCV`
- 📉 Performance evaluated with **classification metrics**
- 🧠 **SHAP explainability** for global + individual predictions
- 🌐 Interactive **Streamlit web app**
- 💼 Real-world business case: Loan Approval System

---

## 📂 Project Structure

loan-approval-with-shap/
├── app.py # Streamlit app
├── model.pkl # Trained model
├── explainer.pkl # SHAP TreeExplainer
├── shap_values.npy # SHAP values for test set
├── X_test.csv # Test features
├── requirements.txt # Python dependencies
└── README.md # Project documentation


🧪 Model Performance
Accuracy: ~85%

Precision, Recall, F1-Score available in the classification report

Class imbalance handled and evaluated properly

🔍 Sample Output
Feature	Impact (SHAP)
Credit_History	Strongly Positive
ApplicantIncome	Slightly Negative
LoanAmount	Negative
Property_Area	Varies by type

📚 Tech Stack
Python

scikit-learn

pandas, numpy

SHAP

Streamlit

joblib, matplotlib

📌 Future Enhancements
Allow users to upload custom applicant data

Add force_plot and other SHAP visualizations

Deploy online with Streamlit Cloud or Render

👨‍💻 Author
Vishal Maheshwary
Aspiring Data Scientist | ML & Analytics Enthusiast
www.linkedin.com/in/vishal-maheshwary
