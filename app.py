"""
Heart Disease Risk Assessment System — by ABXREHMAN
==================================================
Streamlit Web Version with Black, White, Cyan & Red theme.

Required files (in the same directory):
    - knn_heart_model.pkl    (trained classifier)
    - heart_scaler.pkl       (fitted feature scaler)
    - heart_columns.pkl      (expected one-hot-encoded column order)
"""

import os
import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Page Configuration & Styling (Black, White, Cyan & Red Theme)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Assessment System — ABXREHMAN",
    page_icon="🫀",
    layout="centered"
)

# Custom CSS for UI Colors matching your original design
st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #0A0A0A;
            color: #FFFFFF;
        }
        /* Target headers and text */
        h1, h2, h3, p, label {
            color: #FFFFFF !important;
        }
        /* Custom Section Headers */
        .section-header {
            color: #00E5FF !important;
            font-weight: bold;
            font-size: 1.2rem;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }
        /* Footer styling */
        .footer {
            text-align: center;
            color: #8E8E93;
            font-size: 0.8rem;
            margin-top: 3rem;
            border-top: 1px solid #141416;
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

MODEL_FILE = "knn_heart_model.pkl"
SCALER_FILE = "heart_scaler.pkl"
COLUMNS_FILE = "heart_columns.pkl"

# ------------------------------------------------------------------
# Artifact Loading
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    missing = [f for f in (MODEL_FILE, SCALER_FILE, COLUMNS_FILE) if not os.path.exists(f)]
    if missing:
        error_msg = "The following required file(s) were not found:\n\n" + "\n".join(f"  • {name}" for name in missing)
        return None, None, None, error_msg
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        expected_columns = joblib.load(COLUMNS_FILE)
        return model, scaler, expected_columns, None
    except Exception as exc:
        return None, None, None, f"An error occurred while loading model files:\n\n{exc}"

model, scaler, expected_columns, load_error = load_artifacts()

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown("<h1 style='text-align: center;'>🫀 Heart Disease Risk Assessment System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #00E5FF;'>❤️ Machine Learning Diagnostics  •  Single-Page Input Form</p>", unsafe_allow_html=True)

if load_error:
    st.error(load_error)
    st.warning("⚠️ Model files missing — predictions disabled")

# ------------------------------------------------------------------
# Form Inputs
# ------------------------------------------------------------------
st.markdown("<div class='section-header'>👤 1. Patient Demographics</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("🎂 Age (years)", min_value=18, max_value=100, value=40, step=1)
with col2:
    sex = st.selectbox("⚧ Sex", ["M", "F"])
with col3:
    chest_pain = st.selectbox("💔 Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])


st.markdown("<div class='section-header'>🩺 2. Clinical Measurements</div>", unsafe_allow_html=True)

col4, col5 = st.columns(2)
with col4:
    resting_bp = st.number_input("🩸 Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
with col5:
    cholesterol = st.number_input("🧪 Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)

col6, col7 = st.columns(2)
with col6:
    fasting_bs = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dL", ["0", "1"])
with col7:
    max_hr = st.slider("⚡ Max Heart Rate", min_value=60, max_value=220, value=150, step=1)


st.markdown("<div class='section-header'>📈 3. Cardiac Test Results</div>", unsafe_allow_html=True)

col8, col9 = st.columns(2)
with col8:
    resting_ecg = st.selectbox("📊 Resting ECG", ["Normal", "ST", "LVH"])
with col9:
    exercise_angina = st.selectbox("🏃 Exercise-Induced Angina", ["Y", "N"])

col10, col11 = st.columns(2)
with col10:
    oldpeak = st.slider("📉 Oldpeak (ST Depression)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
with col11:
    st_slope = st.selectbox("📐 ST Slope", ["Up", "Flat", "Down"])


# ------------------------------------------------------------------
# Prediction Logic & Action Area
# ------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

# Disable button if models are not loaded successfully
predict_disabled = load_error is not None

if st.button("🚀 Analyze Heart Disease Risk", use_container_width=True, disabled=predict_disabled):
    try:
        # 1. Create a DataFrame from inputs matching your ML model structure
        input_data = pd.DataFrame([{
            'Age': age,
            'Sex': sex,
            'ChestPainType': chest_pain,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': int(fasting_bs),
            'RestingECG': resting_ecg,
            'MaxHR': max_hr,
            'ExerciseAngina': exercise_angina,
            'Oldpeak': oldpeak,
            'ST_Slope': st_slope
        }])

        # 2. Perform One-Hot Encoding matching your model's pipeline
        input_encoded = pd.get_dummies(input_data)
        
        # Reindex columns to match expected model features order
        input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)

        # 3. Scale features using your fitted scaler
        input_scaled = scaler.transform(input_encoded)

        # 4. Predict Risk using your classifier
        prediction = model.predict(input_scaled)[0]
        
        # Try to get prediction probabilities if available for better insight
        try:
            probabilities = model.predict_proba(input_scaled)[0]
            confidence = probabilities[prediction] * 100
            confidence_text = f" (Confidence: {confidence:.1f}%)"
        except AttributeError:
            confidence_text = ""

        # 5. Display Result Card
        st.markdown("<hr style='border-color: #141416;'>", unsafe_allow_html=True)
        if prediction == 1:
            st.error(f"🚨 **HIGH RISK DETECTED**{confidence_text}\nThe assessment indicates warning signs consistent with cardiac disease risk. Please consult a cardiologist.")
        else:
            st.success(f"💚 **LOW RISK / NORMAL**{confidence_text}\nThe clinical markers fall within acceptable baseline ranges. Continue maintaining a healthy lifestyle.")

    except Exception as prediction_error:
        st.error(f"An error occurred during prediction: {prediction_error}")

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown(
    "<div class='footer'>Heart Disease Risk Assessment System — Developed by ABXREHMAN<br>Powered by Streamlit Cloud & Machine Learning</div>", 
    unsafe_allow_html=True
)
