import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Employee Salary Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main {background-color: #1f2937; color: #e5e7eb;}
.header {color: #93c5fd; font-size: 32px; font-weight: 700; text-align: center; margin-bottom: 20px;}
.subheader {color: #d1d5db; font-size: 22px; font-weight: 600; margin-top: 30px; margin-bottom: 10px;}
.info-box {background-color: #1f2937; border-left: 6px solid #3b82f6; padding: 20px; border-radius: 10px; font-size: 16px; line-height: 1.6; color: #f3f4f6; margin-bottom: 30px;}
.stNumberInput input, .stSelectbox div[data-baseweb="select"] {background-color: #374151; color: #e5e7eb; border: 1px solid #4b5563; border-radius: 5px;}
.stSelectbox div[data-baseweb="select"] > div {color: #e5e7eb;}
.center-button {display: flex; justify-content: center; margin-top: 25px;}
.stButton>button {background-color: #3b82f6; color: white; padding: 10px 25px; border-radius: 10px; font-weight: bold; font-size: 16px; border: none;}
.stButton>button:hover {background-color: #60a5fa;}
.prediction-box {background-color: #2563eb; color: white; padding: 25px; font-size: 22px; font-weight: bold; text-align: center; border-radius: 12px; margin-top: 30px; line-height: 1.6;}
.prediction-box h6 {color: #d1d5db; text-align: left; font-size: 14px; margin-top: 10px;}
.metrics-box {background-color: #374151; padding: 15px; border-radius: 8px; margin-top: 15px; text-align: center;}
.metrics-box p {font-size: 16px; color: #f3f4f6;}
</style>
""", unsafe_allow_html=True)

# Load models and data
model_data = joblib.load("all_salary_models.pkl")
models = model_data["models"]
label_encoders = model_data["label_encoders"]
scaler = model_data["scaler"]
feature_names = model_data["feature_names"]

# Hardcode R² and RMSE scores (from training)
model_performance = {
    "Linear Regression": {"R2": 0.7163, "RMSE": 14222.31},
    "Decision Tree": {"R2": 0.7849, "RMSE": 12384.41},
    "Random Forest": {"R2": 0.8756, "RMSE": 9419.67},
    "Gradient Boosting": {"R2": 0.9692, "RMSE": 4690.05},
    "XGBoost": {"R2": 0.9449, "RMSE": 6268.71}
}

# Load comparison charts
comparison_chart_r2 = "images/model_comparison.png"
comparison_chart_rmse = "images/rmse_comparison.png"

# Header
st.markdown('<div class="header">💼 Employee Salary Predictor</div>', unsafe_allow_html=True)

# Info Box
st.markdown(f"""
<div class="info-box">
    <b>📘 Available Models:</b> Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost <br>
    <b>📊 Compare accuracy and choose the best model for your case.</b>
</div>
""", unsafe_allow_html=True)

# Model selection
st.markdown('<div class="subheader">⚙️ Select Prediction Model</div>', unsafe_allow_html=True)
model_choice = st.selectbox("Select Model for Prediction", list(models.keys()))
selected_model = models[model_choice]

# Input Form
st.markdown('<div class="subheader">📝 Enter Employee Details</div>', unsafe_allow_html=True)
with st.form("salary_form"):
    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    gender = st.selectbox("Gender", options=label_encoders["Gender"].classes_)
    education_level = st.selectbox("Education Level", options=label_encoders["Education Level"].classes_)
    job_title = st.selectbox("Job Title", options=label_encoders["Job Title"].classes_)
    years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)

    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    submit_button = st.form_submit_button("Predict Salary")
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction output
if submit_button:
    # Prepare input DataFrame
    input_df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Education Level": [education_level],
        "Job Title": [job_title],
        "Years of Experience": [years_of_experience]
    })

    # Encode categorical variables
    for col in ["Gender", "Education Level", "Job Title"]:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict
    predicted_salary = selected_model.predict(input_scaled)[0]

    # Display prediction
    st.markdown(f"""
    <div class="prediction-box">
        💰 <b>Estimated Annual Salary:</b><br>
        USD ${predicted_salary:,.2f}<br>
        <h6>📌 Based on {model_choice} model.</h6>
    </div>
    """, unsafe_allow_html=True)

    # Show model accuracy metrics
    perf = model_performance[model_choice]
    st.markdown(f"""
    <div class="metrics-box">
        <p>📊 <b>{model_choice} Performance:</b></p>
        <p>✅ R² Score: {perf['R2']:.4f}</p>
        <p>✅ RMSE: {perf['RMSE']:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

# Model comparison charts
st.markdown('<div class="subheader">📊 Model Performance Comparison</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.image(comparison_chart_r2, caption="R² Score Comparison", use_container_width=True)
with col2:
    st.image(comparison_chart_rmse, caption="RMSE Comparison", use_container_width=True)
# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; color: #9ca3af; font-size:14px;">
    <p>© 2023 Employee Salary Predictor. All rights reserved.</p>
    <p>Developed by <strong>Sharavana Ragav</strong></p>
</div>
""", unsafe_allow_html=True)