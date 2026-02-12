import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model/model.pkl")

st.title("🏠 House Price Prediction App")

st.write("Enter house details below:")

area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, step=100)
bedrooms = st.slider("Bedrooms", 1, 6)
bathrooms = st.slider("Bathrooms", 1, 5)

if st.button("Predict Price"):
    features = np.array([[area, bedrooms, bathrooms]])
    prediction = model.predict(features)

    st.success(f"Estimated House Price: ₹ {prediction[0]:,.2f}")
