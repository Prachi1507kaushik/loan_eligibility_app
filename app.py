import streamlit as st
import numpy as np
import pickle
import os

st.title("🏦 Loan Eligibility Prediction System")
st.write("Enter applicant details below:")

# ✅ Correct paths
model_path = "loan_model.pkl"
scaler_path = "scaler.pkl"

# ✅ Check files properly
st.write("Model exists:", os.path.exists(model_path))
st.write("Scaler exists:", os.path.exists(scaler_path))

# ❗ Load model safely
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# INPUT FIELDS
age = st.number_input("Age")
income = st.number_input("Income")
loan_amount = st.number_input("Loan Amount")
credit_score = st.number_input("Credit Score")
Employment_Years=st.number_input("Employment_Years")
Education_level=st.number_input("Education_Level")
Housing_Status_Own=st.number_input("Housing_Status_Own(0/1)")
Housing_Status_Rent=st.number_input("Housing_Status_Rent(0/1)")



# Prediction
if st.button("Check Eligibility"):

    features = np.array([[ age,income, loan_amount, credit_score,Employment_Years,Education_level,Housing_Status_Own,Housing_Status_Rent]])
    features_scaled = scaler.transform(features)

    prob = model.predict_proba(features_scaled)[0][1]

    if prob > 0.3:
        st.success(f"✅ Eligible for Loan (Confidence: {prob:.2f})")
    else:
        st.error(f"❌ Not Eligible (Confidence: {prob:.2f})")