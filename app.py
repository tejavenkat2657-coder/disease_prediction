"""
app.py  —  Streamlit Disease Prediction App
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import subprocess
from pathlib import Path

# ─── Auto-generate dataset + train model if not present ──────────────────────
# This runs automatically on Streamlit Cloud first boot
def setup_model():
    os.makedirs("data",  exist_ok=True)
    os.makedirs("model", exist_ok=True)
    subprocess.run(["python", "generate_dataset.py"], check=True)
    subprocess.run(["python", "train_model.py"],      check=True)

if not os.path.exists("model/best_model.pkl"):
    with st.spinner("⚙️ Setting up ML model for first time... please wait 30 seconds"):
        setup_model()
    st.success("✅ Model ready! Loading app...")
    st.rerun()

# ─── Now safe to import predictor ────────────────────────────────────────────
from predictor import predict, get_symptom_list, get_model_info

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Predictor · Malaria & Dengue",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.hero {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.5rem;
    color: #fff;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero p { margin: 0.4rem 0 0; color: #a8d8ea; font-size: 1rem; }

.chip {
    display: inline-block;
    background: #e8f4fd;
    color: #1565c0;
    border: 1px solid #90caf9;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 3px;
}

.result-card {
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-top: 1rem;
    color: #fff;
}
.result-malaria { background: linear-gradient(135deg, #c0392b, #e74c3c); }
.result-dengue  { background: linear-gradient(135deg, #e67e22, #f39c12); }
.result-card h2 { font-family:'DM Serif Display',serif; font-size:2rem; margin:0; }
.result-card p  { margin:0.3rem 0 0; font-size:1rem; opacity:0.9; }

.conf-badge {
    display: inline-block;
    background: rgba(255,255,255,0.25);
    border-radius: 30px;
    padding: 4px 16px;
    font-weight: 600;
    margin-top: 0.8rem;
    font-size: 1.1rem;
}

.model-badge {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    color: #166534;
}
</style>
""", unsafe_allow_html=True)

# ─── Hero header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🩺 Disease Predictor</h1>
  <p>Malaria &amp; Dengue · Symptom-based ML diagnosis assistant</p>
</div>
""", unsafe_allow_html=True)

# ─── Load model info ──────────────────────────────────────────────────────────
info     = get_model_info()
all_syms = get_symptom_list()

st.markdown(
    f'<div class="model-badge">✅ Active model: <strong>{info["model_name"]}</strong> '
    f'&nbsp;|&nbsp; Accuracy: <strong>{info["accuracy"]*100:.1f}%</strong></div>',
    unsafe_allow_html=True
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Model Comparison")
    cmp_path = Path("model/model_comparison.csv")
    if cmp_path.exists():
        cmp_df = pd.read_csv(cmp_path, index_col=0)
        cmp_df.columns = ["Test Acc", "CV Acc"]
        cmp_df = cmp_df.map(lambda x: f"{x*100:.2f}%")
        st.dataframe(cmp_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    Predicts **Malaria** or **Dengue** from symptoms
    using a machine-learning model trained on a
    clinical symptom dataset.

    **Models trained:**
    - Decision Tree
    - Random Forest
    - Naïve Bayes

    ⚕️ Not a substitute for medical advice.
    """)

# ─── Main input form ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📝 Patient Information")

col_a, col_b = st.columns(2)
with col_a:
    age_group = st.selectbox(
        "Age Group",
        options=["1-10", "10-20", "20-30", "30-40", "40-50", "50-60"],
        index=2,
    )
with col_b:
    gender = st.selectbox("Gender", options=["Male", "Female"])

st.markdown("---")
st.markdown("## 🤒 Select Symptoms")
st.caption("Check all symptoms the patient is currently experiencing.")

DISPLAY = {
    "fever":              "Fever",
    "chills":             "Chills",
    "sweating":           "Sweating",
    "headache":           "Headache",
    "nausea":             "Nausea",
    "vomiting":           "Vomiting",
    "muscle_pain":        "Muscle Pain",
    "fatigue":            "Fatigue",
    "anemia":             "Anemia",
    "spleen_enlargement": "Spleen Enlargement",
    "jaundice":           "Jaundice",
    "high_fever_cycles":  "High Fever Cycles",
    "shivering":          "Shivering",
    "loss_of_appetite":   "Loss of Appetite",
    "dark_urine":         "Dark Urine",
    "severe_headache":    "Severe Headache",
    "pain_behind_eyes":   "Pain Behind Eyes",
    "joint_pain":         "Joint Pain",
    "skin_rash":          "Skin Rash",
    "mild_bleeding":      "Mild Bleeding",
    "low_platelet_count": "Low Platelet Count",
    "abdominal_pain":     "Abdominal Pain",
    "swollen_glands":     "Swollen Glands",
    "dizziness":          "Dizziness",
}

cols    = st.columns(4)
selected = []
for i, sym in enumerate(sorted(all_syms)):
    label   = DISPLAY.get(sym, sym.replace("_", " ").title())
    checked = cols[i % 4].checkbox(label, key=sym)
    if checked:
        selected.append(sym)

# ─── Predict button ───────────────────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔍 Predict Disease", type="primary", use_container_width=True)

if predict_btn:
    if len(selected) < 2:
        st.warning("⚠️ Please select at least 2 symptoms for a meaningful prediction.")
    else:
        with st.spinner("Analysing symptoms…"):
            result = predict(selected, age_group, gender)

        disease = result["disease"]
        conf    = result["confidence"]
        probs   = result["probabilities"]

        card_cls = "result-malaria" if disease == "Malaria" else "result-dengue"
        icon     = "🦟" if disease == "Malaria" else "🦠"

        st.markdown(f"""
        <div class="result-card {card_cls}">
          <h2>{icon} {disease}</h2>
          <p>Based on {len(selected)} reported symptoms · Age {age_group} · {gender}</p>
          <div class="conf-badge">Confidence: {conf*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Probability Breakdown")
        prob_df = pd.DataFrame(
            {"Disease": list(probs.keys()), "Probability": list(probs.values())}
        ).set_index("Disease")
        st.bar_chart(prob_df, use_container_width=True, height=200)

        st.markdown("### Symptoms Reported")
        chips_html = " ".join(
            f'<span class="chip">{DISPLAY.get(s, s.replace("_"," ").title())}</span>'
            for s in selected
        )
        st.markdown(chips_html, unsafe_allow_html=True)

        st.markdown("---")
        if disease == "Malaria":
            st.info("""
**About Malaria** — Caused by *Plasmodium* parasites via *Anopheles* mosquito bites.
Treatments include Artemisinin-based Combination Therapies (ACTs).
Seek immediate medical attention for blood smear confirmation.
            """)
        else:
            st.warning("""
**About Dengue** — Viral infection spread by *Aedes* mosquitoes.
No specific antiviral treatment; management is supportive (fluids, rest, paracetamol).
Monitor platelet counts closely and consult a doctor without delay.
            """)

        st.error("⚕️ **Disclaimer:** This tool is for educational purposes only. Please consult a licensed physician.")

st.markdown("---")
st.caption("Disease Prediction App · Built with Streamlit + scikit-learn · For educational use only")
