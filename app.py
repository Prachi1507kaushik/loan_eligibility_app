import streamlit as st
import numpy as np
import pickle
import os

st.title("🏦 Loan Eligibility Prediction System")
st.write("Enter applicant details below:")

# File paths
model_path = "loan_model.pkl"
scaler_path = "scaler.pkl"

# Check files
st.write("Model exists:", os.path.exists(model_path))
st.write("Scaler exists:", os.path.exists(scaler_path))

# Load model safely
model = None
scaler = None

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    st.success("✅ Model loaded successfully")

except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# INPUT FIELDS
age = st.number_input("Age", min_value=0)
income = st.number_input("Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
credit_score = st.number_input("Credit Score", min_value=0)
employment_years = st.number_input("Employment Years", min_value=0)
education_level = st.number_input("Education Level", min_value=0)
housing_own = st.number_input("Housing Status Own (0/1)", min_value=0, max_value=1)
housing_rent = st.number_input("Housing Status Rent (0/1)", min_value=0, max_value=1)

# Prediction
if st.button("Check Eligibility"):

    if model is None or scaler is None:
        st.error("⚠️ Model not loaded properly")
    else:
        try:
            features = np.array([[ 
                age, income, loan_amount, credit_score,
                employment_years, education_level,
                housing_own, housing_rent
            ]])

            features_scaled = scaler.transform(features)

            prob = model.predict_proba(features_scaled)[0][1]

            if prob > 0.3:
                st.success(f"✅ Eligible for Loan (Confidence: {prob:.2f})")
            else:
                st.error(f"❌ Not Eligible (Confidence: {prob:.2f})")

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
