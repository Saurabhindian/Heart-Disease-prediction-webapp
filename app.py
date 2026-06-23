import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------------- UI STYLE (APPLE LIKE CLEAN UI) ----------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

div[data-testid="stMetric"] {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    padding: 15px;
    border-radius: 15px;
}

.stButton > button {
    border-radius: 12px;
    height: 55px;
    font-size: 18px;
    font-weight: bold;
    background-color: #ff4b4b;
    color: white;
}

.stButton > button:hover {
    background-color: #e63946;
}

.stDataFrame, .stTable {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- LOAD MODEL ----------------
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# ---------------- HEADER ----------------
st.title("❤️ Heart Disease Prediction System")
st.markdown("Predict heart disease risk using Machine Learning + Smart Health Analytics")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Project Info")

st.sidebar.info("""
Developer: **Saurabh Kumar**

Model: KNN Classifier  
Platform: Streamlit Cloud  
Type: ML Healthcare App
""")

st.sidebar.metric("Model Accuracy", "87%")

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)

    sex = st.selectbox("Sex", ["M", "F"])

    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])

    resting_bp = st.number_input("Resting BP", 80, 250, 120)

    cholesterol = st.number_input("Cholesterol", 50, 700, 200)

with col2:
    fasting_bs = st.selectbox("Fasting BS > 120", [0, 1])

    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

    max_hr = st.slider("Max Heart Rate", 60, 220, 150)

    exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])

    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)

    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ---------------- BMI CALCULATOR ----------------
st.subheader("🏃 Lifestyle Information")

c1, c2 = st.columns(2)

with c1:
    height = st.number_input("Height (cm)", 100, 250, 170)

with c2:
    weight = st.number_input("Weight (kg)", 20, 250, 70)

bmi = round(weight / ((height / 100) ** 2), 2)

st.metric("BMI", bmi)

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

    risk_percent = round(probability * 100, 2)
    health_score = round(100 - risk_percent)

    st.divider()
    st.subheader("📊 Prediction Result")

    colA, colB = st.columns(2)

    with colA:
        st.metric("Heart Disease Risk", f"{risk_percent}%")

    with colB:
        st.metric("Heart Health Score", f"{health_score}/100")

    st.progress(int(risk_percent))

    if risk_percent < 30:
        st.success("🟢 Low Risk of Heart Disease")
        st.balloons()

    elif risk_percent < 70:
        st.warning("🟡 Moderate Risk of Heart Disease")

    else:
        st.error("🔴 High Risk of Heart Disease")

    # ---------------- HISTORY ----------------
    st.session_state.history.append(risk_percent)

    st.subheader("📈 Prediction History")
    history_df = pd.DataFrame(st.session_state.history, columns=["Risk"])
    st.line_chart(history_df)

    # ---------------- SUMMARY ----------------
    st.subheader("📝 Patient Summary")

    summary = pd.DataFrame({
        "Parameter": ["Age", "Sex", "BP", "Cholesterol", "Max HR", "BMI"],
        "Value": [age, sex, resting_bp, cholesterol, max_hr, bmi]
    })

    st.table(summary)

    # ---------------- DOWNLOAD REPORT ----------------
    csv = summary.to_csv(index=False)

    st.download_button(
        "📄 Download Report",
        csv,
        "heart_report.csv",
        "text/csv"
    )

    # ---------------- RECOMMENDATIONS ----------------
    st.subheader("💡 Health Recommendations")

    if risk_percent < 30:
        st.success("""
        ✔ Maintain healthy lifestyle  
        ✔ Exercise regularly  
        ✔ Balanced diet  
        ✔ Regular checkups  
        """)

    elif risk_percent < 70:
        st.warning("""
        ⚠ Improve diet  
        ⚠ Reduce cholesterol  
        ⚠ Monitor BP  
        ⚠ Increase exercise  
        """)

    else:
        st.error("""
        🚨 Consult doctor  
        🚨 Regular monitoring  
        🚨 Follow treatment  
        🚨 Avoid smoking/alcohol  
        """)

# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown("""
### ❤️ Heart Disease Prediction System  
Developed by **Saurabh Kumar**

Machine Learning • Streamlit • Healthcare AI

⚠ Educational purpose only
""")
