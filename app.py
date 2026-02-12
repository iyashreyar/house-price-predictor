import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd

# -------------------- Page Config --------------------
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

# -------------------- Load Model --------------------
model_path = os.path.join(os.path.dirname(__file__), "model", "rf_regressor.pkl")
model = joblib.load(model_path)

# -------------------- Title --------------------
st.title("🏠 House Price Prediction App")
st.markdown("""
This application predicts house prices using a tuned **Random Forest Regressor**.
Enter the property details below to get an estimated price.
""")

st.divider()

# -------------------- Input Section --------------------
st.subheader("📌 Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, step=100)
    bedrooms = st.slider("Bedrooms", 1, 10)

with col2:
    bathrooms = st.slider("Bathrooms", 1, 10)

# -------------------- Prediction --------------------
if st.button("Predict Price", use_container_width=True):
    try:
        features = np.array([[area, bedrooms, bathrooms]])
        prediction = model.predict(features)

        st.success(f"💰 Estimated House Price: ₹ {prediction[0]:,.2f}")

        # Feature Importance Section
        st.subheader("🔍 Feature Importance")

        feature_names = ["Area", "Bedrooms", "Bathrooms"]

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        })

        st.bar_chart(importance_df.set_index("Feature"))

    except Exception:
        st.error("⚠️ Something went wrong. Please check inputs.")

st.divider()

# -------------------- Footer --------------------
st.markdown("Built with ❤️ using Streamlit and Scikit-learn")
