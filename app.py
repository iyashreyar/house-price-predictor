import streamlit as st
import joblib
import numpy as np
import os

# Load trained model safely (cloud-friendly path)
model_path = os.path.join(os.path.dirname(__file__), "model", "rf_regressor.pkl")
model = joblib.load(model_path)

st.title("🏠 House Price Prediction App")
st.write("Enter house details below:")

# Better UI Layout
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, step=100)

with col2:
    bedrooms = st.slider("Bedrooms", 1, 10)
    bathrooms = st.slider("Bathrooms", 1, 10)

if st.button("Predict Price"):
    try:
        features = np.array([[area, bedrooms, bathrooms]])
        prediction = model.predict(features)
        st.success(f"Estimated House Price: ₹ {prediction[0]:,.2f}")
    except Exception as e:
        st.error("Something went wrong. Please check inputs.")
