import streamlit as st
import numpy as np
import pickle
import os

# Safe loading
if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
    st.error("Model files missing. Train model first.")
    st.stop()

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Heart Failure Risk Prediction", page_icon="❤️")

st.title("❤️ Heart Failure Risk Prediction System")
st.write("Enter patient clinical details")

st.divider()

# -------------------------
# Input UI
# -------------------------
col1, col2 = st.columns(2)

with col1:

    age = st.number_input("Age", 18, 100, 50)

    anaemia = st.selectbox("Anaemia", ["No", "Yes"])

    creatinine_phosphokinase = st.number_input(
        "Creatinine Phosphokinase",
        20, 8000, 250
    )

    diabetes = st.selectbox("Diabetes", ["No", "Yes"])

    ejection_fraction = st.number_input(
        "Ejection Fraction (%)",
        10, 80, 38
    )

    high_blood_pressure = st.selectbox(
        "High Blood Pressure",
        ["No", "Yes"]
    )

with col2:

    platelets = st.number_input(
        "Platelets",
        25000.0, 850000.0, 250000.0
    )

    serum_creatinine = st.number_input(
        "Serum Creatinine",
        0.1, 10.0, 1.0
    )

    serum_sodium = st.number_input(
        "Serum Sodium",
        110, 150, 137
    )

    sex = st.selectbox("Gender", ["Female", "Male"])

    smoking = st.selectbox("Smoking", ["No", "Yes"])

# -------------------------
# Encoding
# -------------------------
anaemia = 1 if anaemia == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0
high_blood_pressure = 1 if high_blood_pressure == "Yes" else 0
sex = 1 if sex == "Male" else 0
smoking = 1 if smoking == "Yes" else 0

# -------------------------
# Prediction
# -------------------------
st.divider()

if st.button("🔍 Predict Heart Failure Risk"):

    input_data = np.array([[age, anaemia, creatinine_phosphokinase,
                            diabetes, ejection_fraction,
                            high_blood_pressure, platelets,
                            serum_creatinine, serum_sodium,
                            sex, smoking]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Failure")
        st.write(f"Risk Probability: {probability[0][1]*100:.2f}%")
    else:
        st.success("✅ Low Risk of Heart Failure")
        st.write(f"Risk Probability: {probability[0][0]*100:.2f}%")

st.caption("Educational tool only — Not medical diagnosis.")
