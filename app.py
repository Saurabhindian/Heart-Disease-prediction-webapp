import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime
import json
import os
import anthropic

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioSense | Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark medical theme */
    .stApp {
        background: #0d1117;
        color: #e6edf3;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #141824 100%);
        border: 1px solid #21262d;
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #e63946, #ff6b6b, #ffd166);
    }
    .main-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #fff;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #8b949e;
        font-size: 0.95rem;
        margin: 0;
        line-height: 1.5;
    }

    .section-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: #e63946;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid #21262d;
    }

    /* Risk meter */
    .risk-display {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin: 20px 0;
    }
    .risk-percent {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 4.5rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 8px;
    }
    .risk-label {
        font-size: 1rem;
        font-weight: 500;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .low-risk    { color: #3fb950; }
    .mod-risk    { color: #ffd166; }
    .high-risk   { color: #e63946; }

    /* Metric boxes */
    .metric-box {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-box .label {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    .metric-box .value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #e6edf3;
    }

    /* Feature importance bar */
    .feat-bar-wrap { margin: 6px 0; }
    .feat-label    { font-size: 0.82rem; color: #8b949e; margin-bottom: 3px; }
    .feat-bar-bg   { background: #21262d; border-radius: 4px; height: 8px; }
    .feat-bar-fill { background: linear-gradient(90deg, #e63946, #ff6b6b); border-radius: 4px; height: 8px; }

    /* Recommendation cards */
    .rec-card {
        background: #161b22;
        border-left: 3px solid;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-bottom: 10px;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .rec-low  { border-color: #3fb950; }
    .rec-mod  { border-color: #ffd166; }
    .rec-high { border-color: #e63946; }

    /* History table */
    .hist-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 14px;
        border-bottom: 1px solid #21262d;
        font-size: 0.88rem;
    }
    .hist-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d1117 !important;
        border-right: 1px solid #21262d !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #e63946 0%, #c1121f 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 12px 0 !important;
        letter-spacing: 0.3px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88 !important; }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Chatbot styles ── */
    .chat-bubble-user {
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 16px 16px 4px 16px;
        padding: 12px 16px;
        margin: 8px 0 8px 60px;
        color: #e6edf3;
        font-size: 0.9rem;
        line-height: 1.55;
    }
    .chat-bubble-ai {
        background: #1a1f2e;
        border: 1px solid #21262d;
        border-left: 3px solid #e63946;
        border-radius: 4px 16px 16px 16px;
        padding: 12px 16px;
        margin: 8px 60px 8px 0;
        color: #e6edf3;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .chat-avatar {
        font-size: 1.1rem;
        margin-bottom: 3px;
        display: block;
    }
    .chat-context-pill {
        display: inline-block;
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.75rem;
        color: #8b949e;
        margin: 4px 4px 12px 0;
    }
    .chat-context-pill span { color: #e6edf3; font-weight: 500; }
    .quick-q-btn { margin: 4px 4px 4px 0; }
    .typing-dot {
        display: inline-block;
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #e63946;
        animation: blink 1.2s infinite;
        margin: 0 2px;
    }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes blink {
        0%, 80%, 100% { opacity: 0.2; }
        40%            { opacity: 1;   }
    }

    /* Input widgets dark */
    .stSlider > div > div { color: #e6edf3; }
    label { color: #8b949e !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model   = joblib.load("knn_heart_model.pkl")
    scaler  = joblib.load("heart_scaler.pkl")
    columns = joblib.load("heart_columns.pkl")
    return model, scaler, columns

try:
    model, scaler, expected_columns = load_model()
    model_loaded = True
except Exception:
    model_loaded = False

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "last_patient_context" not in st.session_state:
    st.session_state.last_patient_context = None

# ── ANTHROPIC CLIENT ──────────────────────────────────────────────────────────
def get_anthropic_client():
    # Priority: key entered in sidebar UI → environment variable
    api_key = st.session_state.get("anthropic_api_key_input", "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        return anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None

# ── HELPERS ───────────────────────────────────────────────────────────────────
def risk_class(pct):
    if pct < 30:   return "low",  "🟢 Low Risk",   "#3fb950", "low-risk"
    elif pct < 70: return "mod",  "🟡 Moderate Risk","#ffd166","mod-risk"
    else:          return "high", "🔴 High Risk",   "#e63946", "high-risk"


def bmi_category(bmi):
    if bmi < 18.5: return "Underweight"
    elif bmi < 25: return "Normal"
    elif bmi < 30: return "Overweight"
    else:          return "Obese"


def interpret_feature_importance(columns, scaled_row):
    """Approximate per-feature influence using absolute scaled values."""
    vals = np.abs(scaled_row[0])
    pairs = list(zip(columns, vals))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:6]


def radar_chart(values, labels):
    """Return a matplotlib figure — dark-themed radar."""
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles      = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(3.8, 3.8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')

    ax.plot(angles, values_plot, color='#e63946', linewidth=2)
    ax.fill(angles, values_plot, color='#e63946', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='#8b949e', size=8)
    ax.set_yticklabels([])
    ax.spines['polar'].set_color('#21262d')
    ax.grid(color='#21262d', linewidth=0.8)
    plt.tight_layout()
    return fig


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 CardioSense")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🔬 Predict", "🤖 AI Assistant", "📈 History", "ℹ️ About Model"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Developer:** Saurabh Kumar")
    st.markdown("**Model:** K-Nearest Neighbors")
    st.markdown("**Dataset:** Heart Failure Prediction (Kaggle)")

    st.markdown("---")
    if not model_loaded:
        st.error("⚠️ Model files not found. Demo mode active.")

    st.markdown("---")
    st.markdown("**🔑 API Key**")
    api_key_input = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        value=st.session_state.get("anthropic_api_key_input", ""),
        help="Required for the AI Assistant. Get yours at console.anthropic.com",
        label_visibility="collapsed",
    )
    if api_key_input:
        st.session_state["anthropic_api_key_input"] = api_key_input
        st.success("Key saved ✓", icon="✅")
    else:
        st.caption("Paste your key to enable 🤖 AI Assistant")

    st.caption("v3.0 · Final Year Project · 2024")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if "Predict" in page:

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🫀 CardioSense</h1>
        <p>AI-powered cardiovascular risk assessment · Enter patient vitals to generate a risk prediction</p>
    </div>
    """, unsafe_allow_html=True)

    # ── INPUT FORM ─────────────────────────────────────────────────────────
    with st.form("prediction_form"):

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown('<div class="section-title">👤 Demographics</div>', unsafe_allow_html=True)
            age   = st.slider("Age", 18, 100, 45)
            sex   = st.selectbox("Sex", ["M", "F"])
            height = st.number_input("Height (cm)", 140, 220, 170)
            weight = st.number_input("Weight (kg)",  40, 200,  70)

        with col_b:
            st.markdown('<div class="section-title">🩺 Clinical Vitals</div>', unsafe_allow_html=True)
            resting_bp   = st.number_input("Resting BP (mm Hg)", 80, 250, 120)
            cholesterol  = st.number_input("Cholesterol (mg/dL)", 50, 700, 200)
            fasting_bs   = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1],
                                        format_func=lambda x: "Yes" if x else "No")
            max_hr       = st.slider("Max Heart Rate", 60, 220, 150)

        with col_c:
            st.markdown('<div class="section-title">📋 Cardiac Tests</div>', unsafe_allow_html=True)
            chest_pain      = st.selectbox("Chest Pain Type",
                                           ["ASY", "ATA", "NAP", "TA"],
                                           help="ASY=Asymptomatic, ATA=Atypical Angina, NAP=Non-Anginal, TA=Typical Angina")
            resting_ecg     = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
            exercise_angina = st.selectbox("Exercise-Induced Angina", ["N", "Y"],
                                           format_func=lambda x: "Yes" if x=="Y" else "No")
            oldpeak         = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, 0.1)
            st_slope        = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

        # Patient name for report
        patient_name = st.text_input("Patient Name (optional)", placeholder="e.g. John Doe")

        submitted = st.form_submit_button("🔍 Generate Risk Report", use_container_width=True)

    # ── RESULTS ────────────────────────────────────────────────────────────
    if submitted:

        # Build input
        raw_input = {
            'Age': age, 'RestingBP': resting_bp,
            'Cholesterol': cholesterol, 'FastingBS': fasting_bs,
            'MaxHR': max_hr, 'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1,
        }
        input_df = pd.DataFrame([raw_input])

        if model_loaded:
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df   = input_df[expected_columns]
            scaled_row = scaler.transform(input_df)
            prediction = model.predict(scaled_row)[0]
            probability = model.predict_proba(scaled_row)[0][1]
            risk_pct = round(probability * 100, 2)
        else:
            # Demo mode — random plausible value
            np.random.seed(age + resting_bp)
            risk_pct = round(float(np.random.uniform(20, 80)), 2)
            scaled_row = np.zeros((1, 12))
            input_df   = pd.DataFrame([raw_input])

        level, label_text, color, css_class = risk_class(risk_pct)

        # Save to history
        st.session_state.history.append({
            "time": datetime.datetime.now().strftime("%H:%M · %d %b"),
            "name": patient_name or "Anonymous",
            "risk": risk_pct,
            "level": level,
            "age": age, "sex": sex,
            "bp": resting_bp, "chol": cholesterol,
            "maxhr": max_hr,
        })

        # Save full context for AI chatbot
        bmi_ctx = round(weight / ((height / 100) ** 2), 1)
        st.session_state.last_patient_context = {
            "name": patient_name or "the patient",
            "risk_pct": risk_pct,
            "risk_level": level,
            "age": age, "sex": sex,
            "height": height, "weight": weight, "bmi": bmi_ctx,
            "resting_bp": resting_bp, "cholesterol": cholesterol,
            "fasting_bs": fasting_bs, "max_hr": max_hr,
            "chest_pain": chest_pain, "resting_ecg": resting_ecg,
            "exercise_angina": exercise_angina,
            "oldpeak": oldpeak, "st_slope": st_slope,
        }
        # Reset chat so it starts fresh with new patient
        st.session_state.chat_messages = []

        st.markdown("---")

        # ── RISK SCORE ──────────────────────────────────────────────────
        r1, r2 = st.columns([1, 2])

        with r1:
            st.markdown(f"""
            <div class="risk-display">
                <div class="risk-percent {css_class}">{risk_pct}%</div>
                <div class="risk-label" style="color:{color}">{label_text}</div>
                <div style="color:#8b949e;font-size:0.8rem;margin-top:10px;">Heart Disease Risk Score</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            # Quick metrics
            bmi = round(weight / ((height / 100) ** 2), 1)
            m1, m2, m3, m4 = st.columns(4)
            metrics = [
                ("BMI", f"{bmi}", bmi_category(bmi)),
                ("Age Group", "Senior" if age >= 60 else "Middle" if age >= 40 else "Young", ""),
                ("BP Status", "High" if resting_bp > 130 else "Normal", ""),
                ("Chol Status", "High" if cholesterol > 240 else "Borderline" if cholesterol > 200 else "Normal", ""),
            ]
            for col_m, (lbl, val, sub) in zip([m1, m2, m3, m4], metrics):
                with col_m:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="label">{lbl}</div>
                        <div class="value">{val}</div>
                        <div style="font-size:0.72rem;color:#8b949e;margin-top:2px">{sub}</div>
                    </div>""", unsafe_allow_html=True)

            # Risk gauge bar
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#21262d;border-radius:8px;height:14px;margin:8px 0 4px">
                <div style="background:linear-gradient(90deg,#3fb950,#ffd166,#e63946);
                            width:{risk_pct}%;height:14px;border-radius:8px;
                            transition:width .6s ease"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#8b949e">
                <span>0% Low</span><span>30%</span><span>70%</span><span>High 100%</span>
            </div>
            """, unsafe_allow_html=True)

        # ── FEATURE IMPORTANCE + RADAR ───────────────────────────────────
        st.markdown("---")
        fi_col, rad_col = st.columns([3, 2])

        with fi_col:
            st.markdown("### 📊 Top Contributing Factors")
            if model_loaded:
                top_feats = interpret_feature_importance(list(expected_columns), scaled_row)
            else:
                top_feats = [
                    ("Age", 2.1), ("Oldpeak", 1.8), ("Cholesterol", 1.5),
                    ("MaxHR", 1.3), ("RestingBP", 1.1), ("FastingBS", 0.9)
                ]
            max_val = max(v for _, v in top_feats) or 1
            for feat, val in top_feats:
                pct_w = int((val / max_val) * 100)
                clean = feat.replace("_", " ").replace("ChestPainType", "Chest Pain").replace("ExerciseAngina", "Ex. Angina")
                st.markdown(f"""
                <div class="feat-bar-wrap">
                    <div class="feat-label">{clean}</div>
                    <div class="feat-bar-bg">
                        <div class="feat-bar-fill" style="width:{pct_w}%"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        with rad_col:
            st.markdown("### 🕸️ Risk Profile Radar")
            norm = lambda v, lo, hi: min(max((v - lo) / (hi - lo), 0), 1)
            radar_vals = [
                norm(age, 18, 100),
                norm(resting_bp, 80, 250),
                norm(cholesterol, 50, 700),
                norm(max_hr, 60, 220),
                norm(oldpeak, 0, 6),
                1 if exercise_angina == "Y" else 0,
            ]
            radar_labels = ["Age", "BP", "Chol", "MaxHR", "Oldpeak", "Ex.Angina"]
            fig = radar_chart(radar_vals, radar_labels)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

        # ── RECOMMENDATIONS ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 💡 Personalised Recommendations")

        base_recs = {
            "low": [
                ("✔ Maintain a balanced diet rich in vegetables, fruits, and whole grains.", "rec-low"),
                ("✔ Exercise at least 150 min/week of moderate aerobic activity.", "rec-low"),
                ("✔ Schedule a full cardiac checkup annually.", "rec-low"),
                ("✔ Avoid smoking and limit alcohol to 1–2 units/day.", "rec-low"),
            ],
            "mod": [
                ("⚠ Reduce saturated fat and sodium in your diet.", "rec-mod"),
                ("⚠ Monitor blood pressure at home weekly.", "rec-mod"),
                ("⚠ Walk 30 min daily; avoid high-intensity exercise without clearance.", "rec-mod"),
                ("⚠ Consult a physician about cholesterol management.", "rec-mod"),
                ("⚠ Follow up with an ECG test within 3 months.", "rec-mod"),
            ],
            "high": [
                ("🚨 Consult a cardiologist immediately.", "rec-high"),
                ("🚨 Begin medication only under doctor's supervision.", "rec-high"),
                ("🚨 Undergo stress test, echocardiogram, or angiography.", "rec-high"),
                ("🚨 Strictly avoid smoking, alcohol, and high-sodium food.", "rec-high"),
                ("🚨 Monitor heart rate and BP daily; keep an emergency contact ready.", "rec-high"),
            ],
        }

        # Dynamic extras
        extra_recs = []
        if bmi >= 30:
            extra_recs.append(("⚠ BMI indicates obesity — weight loss of 5–10% can significantly reduce cardiac risk.", "rec-mod"))
        if cholesterol > 240:
            extra_recs.append(("⚠ Cholesterol is high — consider plant sterols, oats, and statins (consult doctor).", "rec-mod"))
        if resting_bp > 140:
            extra_recs.append(("⚠ BP is stage-2 hypertensive — reduce sodium intake to <1500 mg/day.", "rec-mod"))
        if fasting_bs == 1:
            extra_recs.append(("⚠ Elevated fasting blood sugar — consider HbA1c testing for diabetes screening.", "rec-mod"))

        all_recs = base_recs[level] + extra_recs
        rec_cols = st.columns(2)
        for i, (text, css) in enumerate(all_recs):
            with rec_cols[i % 2]:
                st.markdown(f'<div class="rec-card {css}">{text}</div>', unsafe_allow_html=True)

        # ── DOWNLOADABLE REPORT ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📄 Download Report")

        report_lines = [
            "=" * 56,
            "       CARDIOSENSE — HEART DISEASE RISK REPORT",
            "=" * 56,
            f"Patient  : {patient_name or 'Anonymous'}",
            f"Date     : {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}",
            "-" * 56,
            "RISK SCORE",
            f"  → {risk_pct}%  ({label_text})",
            "-" * 56,
            "PATIENT PARAMETERS",
            f"  Age             : {age}",
            f"  Sex             : {sex}",
            f"  Height / Weight : {height} cm / {weight} kg  (BMI: {bmi})",
            f"  Resting BP      : {resting_bp} mm Hg",
            f"  Cholesterol     : {cholesterol} mg/dL",
            f"  Fasting BS >120 : {'Yes' if fasting_bs else 'No'}",
            f"  Max Heart Rate  : {max_hr} bpm",
            f"  Chest Pain      : {chest_pain}",
            f"  Resting ECG     : {resting_ecg}",
            f"  Exercise Angina : {exercise_angina}",
            f"  Oldpeak         : {oldpeak}",
            f"  ST Slope        : {st_slope}",
            "-" * 56,
            "RECOMMENDATIONS",
        ]
        for text, _ in all_recs:
            report_lines.append(f"  {text}")
        report_lines += [
            "-" * 56,
            "DISCLAIMER",
            "  This report is AI-generated and not a medical diagnosis.",
            "  Always consult a licensed healthcare professional.",
            "=" * 56,
        ]
        report_text = "\n".join(report_lines)

        st.download_button(
            label="⬇ Download Text Report",
            data=report_text,
            file_name=f"CardioSense_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # Disclaimer
        st.markdown("""
        <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;
                    padding:12px 16px;color:#8b949e;font-size:0.8rem;margin-top:16px">
            ⚠️ <strong>Medical Disclaimer:</strong> This application is a decision-support tool based on
            a machine learning model trained on public datasets. It does not constitute a clinical diagnosis.
            Always seek advice from a qualified healthcare professional before making medical decisions.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
elif "AI Assistant" in page:

    st.markdown("""
    <div class="main-header">
        <h1>🤖 CardioSense AI Assistant</h1>
        <p>Ask anything about the prediction results, heart health, or lifestyle changes</p>
    </div>
    """, unsafe_allow_html=True)

    ctx = st.session_state.last_patient_context

    # ── PATIENT CONTEXT PILLS ─────────────────────────────────────────────
    if ctx:
        lvl_color = {"low": "#3fb950", "mod": "#ffd166", "high": "#e63946"}[ctx["risk_level"]]
        st.markdown(
            f'<div style="margin-bottom:4px;font-size:0.78rem;color:#8b949e;">Active patient context</div>'
            f'<div class="chat-context-pill">Patient <span>{ctx["name"]}</span></div>'
            f'<div class="chat-context-pill">Risk <span style="color:{lvl_color}">{ctx["risk_pct"]}%</span></div>'
            f'<div class="chat-context-pill">Age <span>{ctx["age"]}</span></div>'
            f'<div class="chat-context-pill">Chol <span>{ctx["cholesterol"]} mg/dL</span></div>'
            f'<div class="chat-context-pill">BP <span>{ctx["resting_bp"]} mm Hg</span></div>',
            unsafe_allow_html=True
        )
    else:
        st.info("💡 Run a prediction first (🔬 Predict page) so the assistant has full patient context. You can still ask general heart health questions below.")

    # ── SYSTEM PROMPT BUILDER ─────────────────────────────────────────────
    def build_system_prompt(ctx):
        base = (
            "You are CardioSense Assistant, a helpful, empathetic AI health companion "
            "embedded inside the CardioSense heart disease prediction system. "
            "You explain clinical terms in plain language, help users understand their "
            "risk scores, and provide evidence-based lifestyle advice. "
            "You are supportive and non-alarmist. "
            "Always remind users that you are not a substitute for a real doctor, "
            "but do so briefly and only once per conversation unless they ask about diagnosis. "
            "Keep answers concise (3–5 sentences) unless the user asks for detail. "
            "Use bullet points only when listing multiple items. "
            "Never refuse to explain a term or medical concept. "
            "Do not repeat the patient data back unless it is directly relevant."
        )
        if ctx:
            sex_full = "Male" if ctx["sex"] == "M" else "Female"
            angina_txt = "Yes" if ctx["exercise_angina"] == "Y" else "No"
            fbs_txt = "Yes" if ctx["fasting_bs"] == 1 else "No"
            patient_block = f"""

The following patient data was just assessed by the ML model:
- Name: {ctx['name']}
- Age: {ctx['age']}, Sex: {sex_full}
- BMI: {ctx['bmi']} ({('Underweight' if ctx['bmi']<18.5 else 'Normal' if ctx['bmi']<25 else 'Overweight' if ctx['bmi']<30 else 'Obese')})
- Resting BP: {ctx['resting_bp']} mm Hg
- Cholesterol: {ctx['cholesterol']} mg/dL
- Fasting Blood Sugar >120 mg/dL: {fbs_txt}
- Max Heart Rate: {ctx['max_hr']} bpm
- Chest Pain Type: {ctx['chest_pain']} (ASY=Asymptomatic, ATA=Atypical Angina, NAP=Non-Anginal, TA=Typical)
- Resting ECG: {ctx['resting_ecg']}
- Exercise-Induced Angina: {angina_txt}
- Oldpeak (ST Depression): {ctx['oldpeak']}
- ST Slope: {ctx['st_slope']}
- ML Risk Score: {ctx['risk_pct']}% ({ctx['risk_level'].upper()} risk)

When the user asks about "the result", "my score", "the report", or "the prediction",
refer to these values. Personalise your answers based on this data where relevant."""
            return base + patient_block
        return base

    # ── QUICK QUESTION BUTTONS ────────────────────────────────────────────
    st.markdown("**Quick questions:**")
    quick_qs = []
    if ctx:
        quick_qs = [
            f"What does my {ctx['risk_pct']}% risk score actually mean?",
            f"Why is {ctx['chest_pain']} chest pain type significant?",
            "What lifestyle changes will help me most?",
            f"What does Oldpeak {ctx['oldpeak']} tell the doctor?",
            "Should I be worried? What do I do next?",
            "Explain my ECG result in simple terms.",
        ]
    else:
        quick_qs = [
            "What is heart disease risk?",
            "What does cholesterol have to do with the heart?",
            "How does exercise affect heart health?",
            "What is ST depression on an ECG?",
        ]

    q_cols = st.columns(3)
    for i, q in enumerate(quick_qs):
        with q_cols[i % 3]:
            if st.button(q, key=f"qq_{i}", use_container_width=True):
                st.session_state.chat_messages.append({"role": "user", "content": q})
                st.rerun()

    st.markdown("---")

    # ── RENDER CHAT HISTORY ───────────────────────────────────────────────
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.markdown(
                f'<span class="chat-avatar">🧑</span>'
                f'<div class="chat-bubble-user">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<span class="chat-avatar">🤖</span>'
                f'<div class="chat-bubble-ai">{msg["content"]}</div>',
                unsafe_allow_html=True
            )

    # ── AUTO-RESPOND if last message is from user ─────────────────────────
    needs_response = (
        st.session_state.chat_messages
        and st.session_state.chat_messages[-1]["role"] == "user"
    )

    if needs_response:
        anthropic_client = get_anthropic_client()
        if not anthropic_client:
            # No API key — graceful fallback
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": (
                    "⚠️ **API key not configured.**\n\n"
                    "Paste your Anthropic API key in the **sidebar** (bottom of the left panel) "
                    "and the assistant will activate immediately — no restart needed.\n\n"
                    "Get a free key at [console.anthropic.com](https://console.anthropic.com)."
                )
            })
            st.rerun()
        else:
            with st.spinner(""):
                st.markdown(
                    '<div class="chat-bubble-ai" style="margin-bottom:12px">'
                    '<span class="typing-dot"></span>'
                    '<span class="typing-dot"></span>'
                    '<span class="typing-dot"></span>'
                    '</div>',
                    unsafe_allow_html=True
                )
                try:
                    system_prompt = build_system_prompt(ctx)
                    api_messages  = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_messages
                    ]

                    full_response = ""
                    response_placeholder = st.empty()

                    with anthropic_client.messages.stream(
                        model="claude-sonnet-4-6",
                        max_tokens=600,
                        system=system_prompt,
                        messages=api_messages,
                    ) as stream:
                        for text_chunk in stream.text_stream:
                            full_response += text_chunk
                            response_placeholder.markdown(
                                f'<span class="chat-avatar">🤖</span>'
                                f'<div class="chat-bubble-ai">{full_response}▌</div>',
                                unsafe_allow_html=True
                            )

                    # Final render without cursor
                    response_placeholder.markdown(
                        f'<span class="chat-avatar">🤖</span>'
                        f'<div class="chat-bubble-ai">{full_response}</div>',
                        unsafe_allow_html=True
                    )

                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": full_response
                    })

                except anthropic.AuthenticationError:
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": "❌ Invalid API key. Please check your `ANTHROPIC_API_KEY`."
                    })
                    st.rerun()
                except Exception as e:
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": f"❌ An error occurred: `{e}`"
                    })
                    st.rerun()

    # ── CHAT INPUT ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    user_input = st.chat_input("Ask about the results, symptoms, medications, lifestyle…")
    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input.strip()})
        st.rerun()

    # Clear chat button
    if st.session_state.chat_messages:
        if st.button("🗑 Clear conversation", use_container_width=False):
            st.session_state.chat_messages = []
            st.rerun()

    # Footer note
    st.markdown("""
    <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;
                padding:10px 14px;color:#8b949e;font-size:0.78rem;margin-top:20px">
        🤖 Powered by Claude (Anthropic) · The assistant is aware of the current patient's
        prediction results and provides personalised explanations. This is not a medical diagnosis.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif "History" in page:
    st.markdown("""
    <div class="main-header">
        <h1>📈 Session History</h1>
        <p>All predictions made during this session</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No predictions yet. Go to **🔬 Predict** to get started.")
    else:
        hist = st.session_state.history
        df_hist = pd.DataFrame(hist)

        # Summary stats
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Total Assessments", len(hist))
        with s2:
            high_c = sum(1 for h in hist if h["level"] == "high")
            st.metric("High Risk", high_c)
        with s3:
            avg_r = round(np.mean([h["risk"] for h in hist]), 1)
            st.metric("Avg Risk Score", f"{avg_r}%")
        with s4:
            st.metric("Session Started", hist[0]["time"])

        st.markdown("---")

        # Trend chart
        if len(hist) > 1:
            fig2, ax2 = plt.subplots(figsize=(8, 2.8))
            fig2.patch.set_facecolor('#161b22')
            ax2.set_facecolor('#161b22')
            xs = list(range(len(hist)))
            ys = [h["risk"] for h in hist]
            ax2.plot(xs, ys, color='#e63946', lw=2, marker='o', markersize=6)
            ax2.fill_between(xs, ys, alpha=0.15, color='#e63946')
            ax2.axhline(70, color='#e63946', lw=1, ls='--', alpha=0.4)
            ax2.axhline(30, color='#3fb950', lw=1, ls='--', alpha=0.4)
            ax2.set_xticks(xs)
            ax2.set_xticklabels([h["name"][:8] for h in hist], color='#8b949e', size=8)
            ax2.set_ylabel("Risk %", color='#8b949e', size=9)
            ax2.tick_params(colors='#8b949e')
            ax2.spines[:].set_color('#21262d')
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

        # Table
        st.markdown("#### Patient Records")
        for h in reversed(hist):
            lvl_col = {"low": "#3fb950", "mod": "#ffd166", "high": "#e63946"}[h["level"]]
            st.markdown(f"""
            <div class="hist-row">
                <div class="hist-dot" style="background:{lvl_col}"></div>
                <span style="color:#e6edf3;width:130px">{h['name']}</span>
                <span style="color:#8b949e;width:110px">{h['time']}</span>
                <span style="color:{lvl_col};font-weight:600;width:60px">{h['risk']}%</span>
                <span style="color:#8b949e">Age {h['age']} · {h['sex']} · BP {h['bp']} · Chol {h['chol']}</span>
            </div>""", unsafe_allow_html=True)

        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.rerun()

        # Export
        csv = df_hist.to_csv(index=False)
        st.download_button("⬇ Export as CSV", csv,
                           file_name="cardiosense_history.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ABOUT MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif "About" in page:
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ About the Model</h1>
        <p>Technical details about the CardioSense ML pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Dataset
    The model is trained on the **Heart Failure Prediction Dataset** (Kaggle, 2021), which
    combines five independent heart disease datasets totalling **918 patient records** across
    Cleveland, Hungarian, Switzerland, Long Beach VA, and Stalog sources.

    ### Feature Engineering
    Categorical features (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`)
    were one-hot encoded. Numerical features were standardised using **StandardScaler** before
    being passed to the classifier.

    ### Model: K-Nearest Neighbors
    | Hyperparameter | Value |
    |---|---|
    | Algorithm | KNN |
    | Distance Metric | Euclidean |
    | k (neighbours) | Tuned via cross-validation |
    | Scaling | StandardScaler |

    ### Key Features Used
    | Feature | Clinical Relevance |
    |---|---|
    | Age | Cardiac risk increases significantly after 45 (M) / 55 (F) |
    | Chest Pain Type | Asymptomatic (ASY) is paradoxically high risk |
    | Oldpeak | ST depression > 2 mm strongly correlates with ischaemia |
    | MaxHR | Lower max HR during exercise can indicate poor cardiac output |
    | Exercise Angina | Chest pain during exertion — classic angina indicator |
    | Cholesterol | LDL > 190 mg/dL is a primary modifiable risk factor |
    | Resting BP | Hypertension doubles the risk of heart attack |

    ### Limitations
    - The model should **not** replace clinical judgement.
    - Performance may vary on populations underrepresented in the training data.
    - KNN can be sensitive to feature scaling and irrelevant features.

    ### How to Improve
    - Try **Random Forest** or **XGBoost** for better accuracy and built-in feature importance.
    - Expand the dataset with more recent, diverse patient records.
    - Integrate **SHAP** values for local explainability.
    """)

    st.info("📚 Reference: Fedesoriano (2021). Heart Failure Prediction Dataset. Kaggle.")
