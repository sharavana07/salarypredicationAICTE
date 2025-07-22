import streamlit as st
import pandas as pd
import joblib
import time

# ✅ Page configuration
st.set_page_config(page_title="Employee Salary Predictor", layout="wide", initial_sidebar_state="expanded")

# ✅ Custom CSS for modern dark UI
st.markdown("""
<style>
body {background-color: #111827; color: #e5e7eb;}
header {visibility: hidden;}
.block-container {padding-top: 1rem;}
.stSelectbox div[data-baseweb="select"] {background-color: #1f2937; color: #e5e7eb; border: 1px solid #4b5563; border-radius: 8px;}
.stNumberInput input {background-color: #1f2937; color: #e5e7eb; border: 1px solid #4b5563; border-radius: 8px;}
.stButton>button {background-color: #3b82f6; color: white; font-weight: bold; padding: 12px; border-radius: 10px; width: 100%;}
.stButton>button:hover {background-color: #2563eb;}
.footer {text-align: center; color: #9ca3af; font-size:14px; margin-top: 30px;}
</style>
""", unsafe_allow_html=True)

# ✅ Gradient Header
st.markdown("""
<div style="background: linear-gradient(90deg, #3b82f6, #2563eb); padding: 18px; border-radius: 12px; text-align: center; color: white; font-size: 32px; font-weight: 800;">
💼 Employee Salary Predictor
</div>
""", unsafe_allow_html=True)

# ✅ Sidebar with Model Selection
with st.sidebar:
    st.image("images/salary_icon.png", width=120)
    st.markdown("### ⚙️ Settings")
    st.markdown("Select your prediction model below:")
    
    # Load models and encoders
    model_data = joblib.load("all_salary_models.pkl")
    models = model_data["models"]
    label_encoders = model_data["label_encoders"]
    scaler = model_data["scaler"]

    model_choice = st.selectbox("Prediction Model", list(models.keys()))
    selected_model = models[model_choice]

    st.markdown("---")
    with st.expander("ℹ️ About Models"):
        st.write("""
        - **Linear Regression** – Simple baseline model
        - **Decision Tree** – Handles non-linear data
        - **Random Forest** – Better accuracy, avoids overfitting
        - **Gradient Boosting** – High accuracy
        - **XGBoost** – Best performance
        """)

# ✅ Hardcoded Model Performance
model_performance = {
    "Linear Regression": {"R2": 0.7163, "RMSE": 14222.31},
    "Decision Tree": {"R2": 0.7849, "RMSE": 12384.41},
    "Random Forest": {"R2": 0.8756, "RMSE": 9419.67},
    "Gradient Boosting": {"R2": 0.9692, "RMSE": 4690.05},
    "XGBoost": {"R2": 0.9449, "RMSE": 6268.71}
}

# ✅ Employee Details Input
st.markdown("## 📝 Enter Employee Details")
with st.form("salary_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=30)
        gender = st.selectbox("Gender", options=label_encoders["Gender"].classes_)
        education_level = st.selectbox("Education Level", options=label_encoders["Education Level"].classes_)
    with col2:
        job_title = st.selectbox("Job Title", options=label_encoders["Job Title"].classes_)
        years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)
    
    predict_button = st.form_submit_button("🔍 Predict Salary")

# ✅ Prediction Logic
if predict_button:
    # Show loading animation
    with st.spinner("Calculating salary prediction..."):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        # Prepare DataFrame
        input_df = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Education Level": [education_level],
            "Job Title": [job_title],
            "Years of Experience": [years_of_experience]
        })

        # Encode categorical features
        for col in ["Gender", "Education Level", "Job Title"]:
            input_df[col] = label_encoders[col].transform(input_df[col])

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Predict salary
        predicted_salary = selected_model.predict(input_scaled)[0]

    # ✅ Stylish Prediction Box
    st.markdown(f"""
    <div style="background:#1f2937; padding:25px; border-radius:12px; text-align:center; box-shadow: 0 4px 10px rgba(0,0,0,0.4); margin-top: 25px;">
        <h3 style="color:#60a5fa;">💰 Estimated Annual Salary</h3>
        <h1 style="color:#34d399; font-size:42px; margin:15px 0;">USD ${predicted_salary:,.2f}</h1>
        <p style="color:#9ca3af;">📌 Based on <b>{model_choice}</b> model</p>
    </div>
    """, unsafe_allow_html=True)

    # ✅ Show Model Performance
    perf = model_performance[model_choice]
    st.markdown(f"""
    <div style="background:#374151; padding:15px; border-radius:8px; text-align:center; margin-top:20px;">
        <p style="font-size:16px; color:#f3f4f6;">📊 <b>{model_choice} Performance</b></p>
        <p style="color:#9ca3af;">✅ R² Score: {perf['R2']:.4f}</p>
        <p style="color:#9ca3af;">✅ RMSE: {perf['RMSE']:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

# ✅ Tabs for Model Comparison Charts
st.markdown("## 📊 Model Performance Comparison")
tab1, tab2 = st.tabs(["R² Score", "RMSE"])
with tab1:
    st.image("images/model_comparison.png", use_container_width=True)
with tab2:
    st.image("images/rmse_comparison.png", use_container_width=True)

# ✅ Footer
st.markdown("""
<div class="footer">
    © 2025 Employee Salary Predictor | Developed by <strong>Sharavana Ragav</strong>
</div>
""", unsafe_allow_html=True)
