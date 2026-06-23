import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# ---------------- HEADER ----------------
st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    Predict the likelihood of heart disease using Machine Learning.

    Enter patient health details below and click **Predict Risk**.
    """
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Project Information")

st.sidebar.info("""
Developer: **Saurabh Kumar**

Model: **K-Nearest Neighbors (KNN)**

Deployment: **Streamlit Cloud**

Dataset Features:
- Age
- Blood Pressure
- Cholesterol
- ECG Results
- Chest Pain Type
- Heart Rate
- ST Depression
""")

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)

    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        80, 250, 120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        50, 700, 200
    )

with col2:

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60, 220, 150
    )

    exercise_angina = st.selectbox(
        "Exercise Induced Angina",
        ["Y", "N"]
    )

    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        0.0, 6.0, 1.0
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Risk", use_container_width=True):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,

        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    probability = model.predict_proba(scaled_input)[0][1]

    st.divider()

    st.subheader("📊 Prediction Result")

    risk_percent = round(probability * 100, 2)

    st.metric(
        label="Heart Disease Risk",
        value=f"{risk_percent}%"
    )

    st.progress(int(risk_percent))

    if risk_percent < 30:
        st.success("🟢 Low Risk of Heart Disease")

    elif risk_percent < 70:
        st.warning("🟡 Moderate Risk of Heart Disease")

    else:
        st.error("🔴 High Risk of Heart Disease")
# ---------------- DOCTOR REPORT ----------------

st.subheader("🩺 Medical Assessment Report")

if risk_percent < 30:
    risk_level = "LOW"
elif risk_percent < 70:
    risk_level = "MODERATE"
else:
    risk_level = "HIGH"

report = f"""
PATIENT HEART HEALTH REPORT

----------------------------------------
PATIENT INFORMATION
----------------------------------------
Age               : {age}
Gender            : {sex}

----------------------------------------
VITAL PARAMETERS
----------------------------------------
Blood Pressure    : {resting_bp} mmHg
Cholesterol       : {cholesterol} mg/dL
Maximum Heart Rate: {max_hr} bpm
Fasting Blood Sugar: {fasting_bs}

----------------------------------------
AI ASSESSMENT
----------------------------------------
Risk Percentage   : {risk_percent:.2f}%
Risk Level        : {risk_level}

----------------------------------------
DOCTOR RECOMMENDATIONS
----------------------------------------
"""

if risk_level == "LOW":
    report += """
✓ Maintain a balanced diet
✓ Exercise at least 30 minutes daily
✓ Continue regular health checkups
✓ Maintain healthy body weight
"""

elif risk_level == "MODERATE":
    report += """
⚠ Reduce cholesterol-rich foods
⚠ Monitor blood pressure regularly
⚠ Increase physical activity
⚠ Schedule periodic heart checkups
"""

else:
    report += """
🚨 Consult a cardiologist immediately
🚨 Monitor blood pressure and cholesterol
🚨 Follow a strict heart-healthy diet
🚨 Avoid smoking and alcohol
🚨 Seek professional medical guidance
"""

st.text_area(
    "Generated Medical Report",
    report,
    height=450
)
    # ---------------- SUMMARY ----------------

    st.subheader("📝 Patient Summary")

    summary = pd.DataFrame({
        "Parameter": [
            "Age",
            "Sex",
            "Blood Pressure",
            "Cholesterol",
            "Max Heart Rate"
        ],
        "Value": [
            age,
            sex,
            resting_bp,
            cholesterol,
            max_hr
        ]
    })

    st.table(summary)

    # ---------------- RECOMMENDATIONS ----------------

    st.subheader("💡 Health Recommendations")

    if risk_percent < 30:

        st.success("""
        ✔ Maintain a healthy lifestyle

        ✔ Exercise regularly

        ✔ Continue balanced nutrition

        ✔ Schedule routine health checkups
        """)

    elif risk_percent < 70:

        st.warning("""
        ⚠ Improve dietary habits

        ⚠ Reduce cholesterol intake

        ⚠ Monitor blood pressure regularly

        ⚠ Increase physical activity
        """)

    else:

        st.error("""
        🚨 Consult a cardiologist

        🚨 Monitor heart health frequently

        🚨 Follow prescribed medical advice

        🚨 Avoid smoking and excessive alcohol
        """)

# ---------------- DISCLAIMER ----------------

st.divider()

st.caption(
    """
    Disclaimer:
    This application provides predictions using a machine learning model
    and should not be considered a medical diagnosis.
    Always consult a healthcare professional.
    """
)
