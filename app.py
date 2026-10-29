import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SentinelXAI Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PREMIUM TYPOGRAPHY & GUIDED PASTEL CSS
# ============================================================

st.markdown(
    """
    <style>
        /* Import Plus Jakarta Sans & JetBrains Mono */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        /* The app uses a light canvas; force Streamlit widgets onto the same light palette. */
        :root {
            color-scheme: light !important;
            --text-color: #334155;
            --background-color: #f7f9fd;
            --secondary-background-color: #ffffff;
            --border-color: #e2e8f0;
            --primary-color: #7c3aed;
        }

        html,
        body,
        .stApp,
        [data-testid="stAppViewContainer"] {
            color-scheme: light !important;
            color: #334155;
        }

        /* Gentle Ambient Pastel Gradient Background */
        .stApp {
            background-color: #f7f9fd !important;
            background-image:
                radial-gradient(at 0% 0%, rgba(238, 242, 255, 0.75) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(243, 232, 255, 0.7) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(236, 253, 245, 0.6) 0px, transparent 50%),
                radial-gradient(at 0% 100%, rgba(254, 243, 199, 0.4) 0px, transparent 50%) !important;
            background-attachment: fixed !important;
        }

        /* Keep Streamlit's sidebar toggle, but remove the dark deploy toolbar band. */
        header[data-testid="stHeader"] {
            background: #f7f9fd !important;
        }

        div[data-testid="stToolbar"] {
            background: #f7f9fd !important;
            color: #334155 !important;
        }

        div[data-testid="stToolbar"] button {
            color: #334155 !important;
        }

        .stTabs [data-baseweb="tab"],
        .stTabs [data-baseweb="tab"] * {
            color: #64748b !important;
            opacity: 1 !important;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"],
        .stTabs [data-baseweb="tab"][aria-selected="true"] * {
            color: #7c3aed !important;
        }

        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] *,
        div[data-testid="stMetricDelta"],
        div[data-testid="stMetricDelta"] * {
            opacity: 1 !important;
        }

        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] * {
            color: #64748b !important;
        }

        div[data-testid="stMetricDelta"],
        div[data-testid="stMetricDelta"] * {
            color: #334155 !important;
        }

        div[data-testid="stMain"] [data-testid="stMarkdownContainer"] p:not([style]) {
            color: #334155 !important;
        }

        div[data-testid="stCaptionContainer"],
        div[data-testid="stCaptionContainer"] * {
            color: #64748b !important;
        }

        .js-plotly-plot .xtick text,
        .js-plotly-plot .ytick text,
        .js-plotly-plot .legend text,
        .js-plotly-plot .gtitle,
        .js-plotly-plot .g-xtitle text,
        .js-plotly-plot .g-ytitle text {
            fill: #64748b !important;
            color: #64748b !important;
        }

        [data-testid="stDataFrame"] {
            color-scheme: light !important;
        }
        }

        [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: #334155 !important;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #64748b !important;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
            color: #0f172a !important;
        }

        section[data-testid="stMain"] > div[data-testid="stMainBlockContainer"] {
            padding-top: 1rem !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.96) 0%, rgba(248, 250, 252, 0.96) 100%) !important;
            border-right: 1px solid #e2e8f0 !important;
        }

        /* Safe Typography Application - Targets text elements without clobbering Material Icons */
        html, body, p, h1, h2, h3, h4, h5, h6, label, div.stMarkdown, div.stSelectbox, div.stRadio, div.stTextInput, input, select, textarea {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            letter-spacing: -0.01em;
        }

        /* Slightly enlarge supporting copy while leaving the project title size alone. */
        div.stMarkdown [style*="font-size: 0.72rem"] { font-size: 0.76rem !important; }
        div.stMarkdown [style*="font-size: 0.74rem"] { font-size: 0.78rem !important; }
        div.stMarkdown [style*="font-size: 0.76rem"] { font-size: 0.80rem !important; }
        div.stMarkdown [style*="font-size: 0.78rem"] { font-size: 0.82rem !important; }
        div.stMarkdown [style*="font-size: 0.8rem"] { font-size: 0.84rem !important; }
        div.stMarkdown [style*="font-size: 0.82rem"] { font-size: 0.86rem !important; }
        div.stMarkdown [style*="font-size: 0.84rem"] { font-size: 0.88rem !important; }
        div.stMarkdown [style*="font-size: 0.85rem"] { font-size: 0.90rem !important; }
        div.stMarkdown [style*="font-size: 0.88rem"] { font-size: 0.92rem !important; }
        div.stMarkdown [style*="font-size: 0.9rem"] { font-size: 0.94rem !important; }
        div.stMarkdown [style*="font-size: 0.92rem"] { font-size: 0.96rem !important; }
        div.stMarkdown [style*="font-size: 0.95rem"] { font-size: 0.99rem !important; }

        div.stButton > button {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        }

        .mono {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* CRITICAL: Protect Streamlit Material Icons from font overrides */
        [data-testid="stIcon"],
        [data-testid="stIconMaterial"],
        [data-testid*="Icon"],
        [class*="material-symbols"],
        [class*="material-icons"],
        .material-symbols-rounded,
        .material-symbols-outlined,
        .material-icons,
        button[kind="header"] span,
        summary span[class*="st-"] {
            font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
            font-style: normal !important;
            font-weight: normal !important;
            line-height: 1 !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            display: inline-block !important;
            white-space: nowrap !important;
            word-wrap: normal !important;
            direction: ltr !important;
            -webkit-font-feature-settings: 'liga' 1 !important;
            font-feature-settings: 'liga' 1 !important;
        }

        /* Top Guide Header */
        .header-box {
            background: rgba(255, 255, 255, 0.92);
            backdrop-filter: blur(8px);
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 1.4rem 1.8rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
        }

        .header-title {
            font-size: 1.95rem;
            font-weight: 800;
            color: #0f172a !important;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.65rem;
            letter-spacing: -0.025em;
        }

        .header-desc {
            font-size: 0.98rem;
            color: #64748b;
            margin-top: 0.35rem;
            margin-bottom: 0;
            line-height: 1.5;
        }

        /* 3-Step Stepper Cards */
        .step-container {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(8px);
            border: 1px solid #e9d5ff;
            border-radius: 18px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 14px -2px rgba(124, 58, 237, 0.04);
        }

        .step-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 26px;
            height: 26px;
            border-radius: 50%;
            font-size: 0.76rem;
            font-weight: 700;
            color: #ffffff;
            background: #7c3aed;
            margin-right: 0.5rem;
        }

        /* Soft Pastel Badges & Status */
        .badge-rose {
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #fecaca;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
        }

        .badge-mint {
            background-color: #dcfce7;
            color: #166534;
            border: 1px solid #bbf7d0;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
        }

        .badge-lavender {
            background-color: #f3e8ff;
            color: #6b21a8;
            border: 1px solid #e9d5ff;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
        }

        .badge-peach {
            background-color: #ffedd5;
            color: #9a3412;
            border: 1px solid #fed7aa;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
        }

        /* Streamlined White Cards with Subtle Shadow */
        .card-white {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 1.3rem 1.5rem;
            box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.04);
            margin-bottom: 1.2rem;
        }

        /* Streamlit Native Tab Bar Overrides */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.3rem;
        }

        .stTabs [data-baseweb="tab"] {
            font-size: 0.99rem;
            font-weight: 700;
            color: #64748b;
            padding: 0.5rem 0.2rem 0.8rem 0.2rem;
            border: none;
            background: transparent;
        }

        .stTabs [aria-selected="true"] {
            color: #7c3aed !important;
            border-bottom: 2.5px solid #7c3aed !important;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 0.9rem 1.1rem;
            border-radius: 14px;
            box-shadow: 0 2px 8px -2px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stMetricLabel"] {
            color: #64748b;
            font-weight: 600;
            font-size: 0.86rem;
        }

        div[data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 800;
            font-size: 1.5rem;
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Custom Scenario Buttons */
        div.stButton > button {
            border-radius: 14px;
            font-size: 0.88rem;
            font-weight: 700;
            border: 1.5px solid #e2e8f0;
            background-color: #ffffff;
            color: #1e293b;
            padding: 0.65rem 1rem;
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(15, 23, 42, 0.03);
            text-align: left;
        }

        div.stButton > button:hover {
            border-color: #7c3aed;
            color: #7c3aed;
            background-color: #faf5ff;
            transform: translateY(-1px);
            box-shadow: 0 4px 10px rgba(124, 58, 237, 0.1);
        }

        div.stButton > button[kind="primary"] {
            background-color: #7c3aed !important;
            border-color: #6d28d9 !important;
            color: #ffffff !important;
            box-shadow: 0 4px 12px rgba(124, 58, 237, 0.22) !important;
        }

        /* Expander Clean Border and Padding */
        div[data-testid="stExpander"] {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 16px !important;
            box-shadow: 0 2px 10px -2px rgba(15, 23, 42, 0.04) !important;
            margin-top: 1rem !important;
            overflow: hidden !important;
        }

        div[data-testid="stExpander"] summary {
            padding: 0.75rem 1.1rem !important;
            border-radius: 16px !important;
        }

        div[data-testid="stExpander"] summary p {
            margin: 0 !important;
            font-weight: 700 !important;
            color: #334155 !important;
            padding-left: 0.5rem !important;
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# PROJECT PATHS & DATA SETUP
# ============================================================

PROJECT_DIR = Path(r"C:\Users\Isha Kale\SentinelXAI")
MODEL_PATH = PROJECT_DIR / "models" / "baseline_lstm.keras"
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"

X_TEST_PATH = DATA_DIR / "X_test_lstm.npy"
Y_TEST_PATH = DATA_DIR / "y_test_lstm.npy"

X_FGSM_PATH = DATA_DIR / "X_test_fgsm_eps001.npy"
X_PGD_PATH = DATA_DIR / "X_test_pgd_eps001.npy"
X_DEEPFOOL_PATH = DATA_DIR / "X_test_deepfool.npy"

SHAP_CLEAN_PATH = RESULTS_DIR / "shap_clean_500.npy"
SHAP_FGSM_PATH = RESULTS_DIR / "shap_fgsm_500.npy"
SHAP_PGD_PATH = RESULTS_DIR / "shap_pgd_500.npy"
SHAP_DEEPFOOL_PATH = RESULTS_DIR / "shap_deepfool_500.npy"

BEHAVIOR_PATH = RESULTS_DIR / "xai_behavior_features_500.csv"
FEATURE_NAMES_PATH = DATA_DIR / "feature_names.json"

# ============================================================
# DATA CACHING & LOADERS
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_arrays():
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    X_fgsm = np.load(X_FGSM_PATH)
    X_pgd = np.load(X_PGD_PATH)
    X_deepfool = np.load(X_DEEPFOOL_PATH)

    shap_clean = np.load(SHAP_CLEAN_PATH)
    shap_fgsm = np.load(SHAP_FGSM_PATH)
    shap_pgd = np.load(SHAP_PGD_PATH)
    shap_deepfool = np.load(SHAP_DEEPFOOL_PATH)

    behavior = pd.read_csv(BEHAVIOR_PATH)

    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    return (
        X_test,
        y_test,
        X_fgsm,
        X_pgd,
        X_deepfool,
        shap_clean,
        shap_fgsm,
        shap_pgd,
        shap_deepfool,
        behavior,
        feature_names
    )

try:
    model = load_model()
    (
        X_test,
        y_test,
        X_fgsm,
        X_pgd,
        X_deepfool,
        shap_clean,
        shap_fgsm,
        shap_pgd,
        shap_deepfool,
        behavior,
        feature_names
    ) = load_arrays()
except Exception as e:
    st.error("⚠️ Unable to load SentinelXAI dataset or model files. Check local paths.")
    st.exception(e)
    st.stop()

ATTACK_RESULTS = {
    "FGSM": {
        "accuracy": 0.8878,
        "drift": 0.0158611501,
        "similarity": 0.67960357,
        "overlap": 0.6774,
        "confidence_drop": 0.023536352,
        "severity": "Moderate",
        "name": "Fast Gradient Sign Method",
        "description": "A single mathematical gradient step tweaking packet lengths."
    },
    "PGD": {
        "accuracy": 0.8783,
        "drift": 0.0162214949,
        "similarity": 0.66450081,
        "overlap": 0.6776,
        "confidence_drop": 0.026820794,
        "severity": "Moderate",
        "name": "Projected Gradient Descent",
        "description": "Multi-step iterative attack searching for optimal noise."
    },
    "DeepFool": {
        "accuracy": 0.0537,
        "drift": 0.0277837812,
        "similarity": 0.36024411,
        "overlap": 0.4494,
        "confidence_drop": 0.6168139,
        "severity": "Critical",
        "name": "DeepFool Non-Linear Perturbation",
        "description": "Finds the closest mathematical path to push traffic across the decision boundary."
    }
}

ATTACK_ARRAYS = {
    "FGSM": X_fgsm,
    "PGD": X_pgd,
    "DeepFool": X_deepfool
}

ATTACK_SHAP = {
    "FGSM": shap_fgsm,
    "PGD": shap_pgd,
    "DeepFool": shap_deepfool
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def class_name(label):
    return "ATTACK" if int(label) == 1 else "BENIGN"

def get_prediction(sample):
    probability = float(model.predict(sample, verbose=0)[0][0])
    prediction = 1 if probability >= 0.5 else 0
    confidence = probability if prediction == 1 else (1 - probability)
    return prediction, confidence, probability

def get_status(clean_prediction, adversarial_prediction, actual_class):
    clean_correct = (clean_prediction == actual_class)
    adversarial_correct = (adversarial_prediction == actual_class)
    if clean_correct and not adversarial_correct:
        return "success"
    if clean_correct and adversarial_correct:
        return "robust"
    return "clean_wrong"

def calculate_sample_metrics(clean_shap, adversarial_shap):
    drift = float(np.mean(np.abs(clean_shap - adversarial_shap)))
    clean_norm = np.linalg.norm(clean_shap)
    adversarial_norm = np.linalg.norm(adversarial_shap)
    if clean_norm == 0 or adversarial_norm == 0:
        similarity = 0.0
    else:
        similarity = float(np.dot(clean_shap, adversarial_shap) / (clean_norm * adversarial_norm))

    clean_top10 = set(np.argsort(np.abs(clean_shap))[-10:])
    adversarial_top10 = set(np.argsort(np.abs(adversarial_shap))[-10:])
    overlap = len(clean_top10.intersection(adversarial_top10)) / 10.0
    return drift, similarity, overlap

@st.cache_data
def calculate_global_importance(clean_shap, fgsm_shap, pgd_shap, deepfool_shap, feature_names):
    return pd.DataFrame({
        "Feature": feature_names,
        "Clean": np.mean(np.abs(clean_shap), axis=0),
        "FGSM": np.mean(np.abs(fgsm_shap), axis=0),
        "PGD": np.mean(np.abs(pgd_shap), axis=0),
        "DeepFool": np.mean(np.abs(deepfool_shap), axis=0)
    })

global_importance = calculate_global_importance(
    shap_clean, shap_fgsm, shap_pgd, shap_deepfool, feature_names
)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "sample_number" not in st.session_state:
    st.session_state["sample_number"] = 469
if "attack_name" not in st.session_state:
    st.session_state["attack_name"] = "DeepFool"
if "view_mode" not in st.session_state:
    st.session_state["view_mode"] = "🔰 Beginner Walkthrough"

# Derive the active mode once per Streamlit rerun. The mode toggle buttons
# update this session-state value and then rerun the app.
is_analyst = st.session_state["view_mode"] == "🔬 Analyst Deep-Dive"

# ============================================================
# HEADER & VALUE PROPOSITION
# ============================================================

st.markdown(
    """
    <div class="header-box">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
            <div>
                <h1 class="header-title">
                    🛡️ SentinelXAI <span class="badge-lavender mono" style="font-size: 0.75rem;">INTERACTIVE CONSOLE</span>
                </h1>
                <p class="header-desc">
                    Understand how hackers manipulate network traffic to deceive AI Intrusion Detection Systems (IDS) — and how Explainable AI (XAI) exposes the trick.
                </p>
            </div>
            <div style="display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap;">
                <span class="badge-mint">● SYSTEM READY</span>
                <span class="badge-lavender mono">CICIDS2017 LSTM (95.04% ACC)</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 3-STEP FIRST-TIME USER STEPPER BAR
# ============================================================

st.markdown(
    """
    <div class="step-container">
        <div style="font-size: 0.8rem; font-weight: 800; color: #581c87; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.6rem;">
            🧭 How to Navigate (3-Step Guide)
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 0.75rem;">
            <div style="background: #ffffff; border: 1px solid #e9d5ff; border-radius: 12px; padding: 0.7rem 0.9rem; display: flex; align-items: center; gap: 0.6rem;">
                <span class="step-pill">1</span>
                <div>
                    <div style="font-size: 0.8rem; font-weight: 700; color: #0f172a;">Select a Scenario</div>
                    <div style="font-size: 0.72rem; color: #64748b;">Click an attack scenario card below</div>
                </div>
            </div>
            <div style="background: #ffffff; border: 1px solid #e9d5ff; border-radius: 12px; padding: 0.7rem 0.9rem; display: flex; align-items: center; gap: 0.6rem;">
                <span class="step-pill" style="background: #0284c7;">2</span>
                <div>
                    <div style="font-size: 0.8rem; font-weight: 700; color: #0f172a;">Inspect the Packet Story</div>
                    <div style="font-size: 0.72rem; color: #64748b;">See if the AI caught or was fooled</div>
                </div>
            </div>
            <div style="background: #ffffff; border: 1px solid #e9d5ff; border-radius: 12px; padding: 0.7rem 0.9rem; display: flex; align-items: center; gap: 0.6rem;">
                <span class="step-pill" style="background: #16a34a;">3</span>
                <div>
                    <div style="font-size: 0.8rem; font-weight: 700; color: #0f172a;">Review XAI & Defense</div>
                    <div style="font-size: 0.72rem; color: #64748b;">See what changed & how to fix it</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# STEP 1: VISUAL SCENARIO CARDS (SELF-EXPLANATORY)
# ============================================================

st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
        <span style="font-size: 0.88rem; font-weight: 800; color: #0f172a;">
            👉 Step 1: Select a Traffic Scenario to Test
        </span>
        <span style="font-size: 0.75rem; color: #64748b;">
            Click any scenario to immediately load its data
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

s_col1, s_col2, s_col3, s_col4 = st.columns(4)

with s_col1:
    is_active = (st.session_state["sample_number"] == 469 and st.session_state["attack_name"] == "DeepFool")
    if st.button("🚨 Stealth Attack (DeepFool)\nSample #469 • Major Breach", use_container_width=True, type="primary" if is_active else "secondary", key="scenario_stealth"):
        st.session_state["sample_number"] = 469
        st.session_state["attack_name"] = "DeepFool"
        st.rerun()

with s_col2:
    is_active = (st.session_state["sample_number"] == 31 and st.session_state["attack_name"] == "FGSM")
    if st.button("⚡ Fast Attack (FGSM)\nSample #31 • Gradient Push", use_container_width=True, type="primary" if is_active else "secondary", key="scenario_fast"):
        st.session_state["sample_number"] = 31
        st.session_state["attack_name"] = "FGSM"
        st.rerun()

with s_col3:
    is_active = (st.session_state["sample_number"] == 0 and st.session_state["attack_name"] == "PGD")
    if st.button("🛡️ Defended Attack (PGD)\nSample #0 • Model Caught It", use_container_width=True, type="primary" if is_active else "secondary", key="scenario_defended"):
        st.session_state["sample_number"] = 0
        st.session_state["attack_name"] = "PGD"
        st.rerun()

with s_col4:
    is_active = (st.session_state["sample_number"] == 1 and st.session_state["attack_name"] == "FGSM")
    if st.button("🟢 Clean Normal Traffic\nSample #1 • Safe Web Flow", use_container_width=True, type="primary" if is_active else "secondary", key="scenario_clean"):
        st.session_state["sample_number"] = 1
        st.session_state["attack_name"] = "FGSM"
        st.rerun()

# ============================================================
# SIDEBAR CONTROLS (ADVANCED USERS)
# ============================================================

st.sidebar.markdown(
    """
    <div style="padding-bottom: 0.8rem; border-bottom: 1px solid #e2e8f0; margin-bottom: 1rem;">
        <h3 style="margin: 0; font-size: 1.1rem; color: #0f172a; font-weight: 800;">
            ⚙️ Console Settings
        </h3>
        <p style="margin: 0.2rem 0 0 0; font-size: 0.8rem; color: #64748b;">
            Customize view & sample inspection
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("**Manual Flow Exploration**")

sample_number = st.sidebar.number_input(
    "SHAP Sample Index (0 - 499)",
    min_value=0,
    max_value=len(behavior) - 1,
    value=st.session_state["sample_number"],
    step=1
)
if sample_number != st.session_state["sample_number"]:
    st.session_state["sample_number"] = sample_number

attack_name = st.sidebar.selectbox(
    "Adversarial Attack Vector",
    ["FGSM", "PGD", "DeepFool"],
    index=["FGSM", "PGD", "DeepFool"].index(st.session_state["attack_name"])
)
if attack_name != st.session_state["attack_name"]:
    st.session_state["attack_name"] = attack_name

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style="font-size: 0.78rem; color: #64748b; line-height: 1.6;">
        <p><strong>Active Flow:</strong> #{st.session_state['sample_number']}</p>
        <p><strong>Selected Vector:</strong> {st.session_state['attack_name']}</p>
        <p><strong>Total Pool:</strong> 45,000 flows</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD CURRENT FLOW DATA & COMPUTE
# ============================================================

shap_row = int(st.session_state["sample_number"])
original_index = int(behavior.iloc[shap_row]["sample_index"])

clean_sample = X_test[original_index:original_index + 1]
adversarial_sample = ATTACK_ARRAYS[st.session_state["attack_name"]][original_index:original_index + 1]

clean_shap = shap_clean[shap_row]
adversarial_shap = ATTACK_SHAP[st.session_state["attack_name"]][shap_row]
actual_class = int(y_test[original_index])

clean_prediction, clean_confidence, clean_probability = get_prediction(clean_sample)
adversarial_prediction, adversarial_confidence, adversarial_probability = get_prediction(adversarial_sample)
status = get_status(clean_prediction, adversarial_prediction, actual_class)

local_drift, local_similarity, local_overlap = calculate_sample_metrics(clean_shap, adversarial_shap)

clean_top = set(np.argsort(np.abs(clean_shap))[-5:])
attack_top = set(np.argsort(np.abs(adversarial_shap))[-5:])
added_features = [feature_names[i] for i in attack_top - clean_top][:3]
removed_features = [feature_names[i] for i in clean_top - attack_top][:3]

# ============================================================
# ON-PAGE MODE BANNER & SWITCHER
# ============================================================

m_col1, m_col2 = st.columns([3, 1.2])

with m_col1:
    if not is_analyst:
        st.markdown(
            """
            <div style="background: rgba(243, 232, 255, 0.65); border: 1.5px solid #e9d5ff; border-radius: 14px; padding: 0.65rem 1rem; display: flex; align-items: center; gap: 0.65rem;">
                <span style="font-size: 1.15rem;">🔰</span>
                <div>
                    <span style="font-size: 0.84rem; font-weight: 800; color: #6b21a8;">ACTIVE VIEW: BEGINNER WALKTHROUGH</span>
                    <div style="font-size: 0.74rem; color: #64748b;">Showing narrative-driven incident flow with zero mathematical jargon.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background: rgba(220, 252, 231, 0.65); border: 1.5px solid #bbf7d0; border-radius: 14px; padding: 0.65rem 1rem; display: flex; align-items: center; gap: 0.65rem;">
                <span style="font-size: 1.15rem;">🔬</span>
                <div>
                    <span style="font-size: 0.84rem; font-weight: 800; color: #166534;">ACTIVE VIEW: ANALYST DEEP-DIVE</span>
                    <div style="font-size: 0.74rem; color: #64748b;">All 3 workspaces, 78-feature SHAP matrix, and model resilience telemetry unlocked.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

with m_col2:
    if not is_analyst:
        if st.button("🔬 Switch to Analyst Mode", use_container_width=True, key="btn_toggle_to_analyst"):
            st.session_state["view_mode"] = "🔬 Analyst Deep-Dive"
            st.rerun()
    else:
        if st.button("🔰 Switch to Beginner Mode", use_container_width=True, key="btn_toggle_to_beginner"):
            st.session_state["view_mode"] = "🔰 Beginner Walkthrough"
            st.rerun()

st.markdown('<div style="height: 0.6rem;"></div>', unsafe_allow_html=True)

# ============================================================
# BRANCH 1: BEGINNER WALKTHROUGH (ZERO JARGON, SINGLE-PAGE NARRATIVE)
# ============================================================

if not is_analyst:
    # 1. Plain-English Incident Banner
    if status == "success" and actual_class == 1:
        st.markdown(
            f"""
            <div style="background: #fef2f2; border: 1.5px solid #fecaca; border-radius: 18px; padding: 1.3rem 1.6rem; margin-bottom: 1.2rem;">
                <div style="display: flex; align-items: flex-start; gap: 1rem;">
                    <div style="font-size: 2rem; background: #fee2e2; border-radius: 14px; padding: 0.5rem 0.75rem; border: 1px solid #fca5a5;">
                        ⚠️
                    </div>
                    <div>
                        <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                            <span class="badge-rose mono">BREACH DETECTED</span>
                            <span style="font-size: 1.05rem; font-weight: 800; color: #991b1b;">
                                AI Deceived: Malicious Traffic Disguised as Harmless
                            </span>
                        </div>
                        <p style="font-size: 0.88rem; color: #475569; margin-top: 0.45rem; line-height: 1.6; margin-bottom: 0;">
                            This network flow is an actual cyber attack. Without modification, the intrusion detection model correctly flagged it as <strong>{class_name(clean_prediction)} ({clean_confidence:.1%} confidence)</strong>.
                            However, the attacker subtly tweaked packet length and timing statistics using <strong>{st.session_state['attack_name']}</strong>.
                            The AI was tricked into reclassifying it as safe <strong>{class_name(adversarial_prediction)} ({adversarial_confidence:.1%} confidence)</strong>, allowing the threat to pass undetected!
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif status == "success" and actual_class == 0:
        st.markdown(
            f"""
            <div style="background: #fff7ed; border: 1.5px solid #fed7aa; border-radius: 18px; padding: 1.3rem 1.6rem; margin-bottom: 1.2rem;">
                <div style="display: flex; align-items: flex-start; gap: 1rem;">
                    <div style="font-size: 2rem; background: #ffedd5; border-radius: 14px; padding: 0.5rem 0.75rem; border: 1px solid #fdba74;">ℹ️</div>
                    <div>
                        <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                            <span class="badge-peach mono">FALSE ALARM</span>
                            <span style="font-size: 1.05rem; font-weight: 800; color: #9a3412;">Perturbation Caused a Benign Flow to Be Flagged</span>
                        </div>
                        <p style="font-size: 0.88rem; color: #475569; margin-top: 0.45rem; line-height: 1.6; margin-bottom: 0;">
                            The ground-truth label is BENIGN. The model changed its prediction from BENIGN to ATTACK after the {st.session_state['attack_name']} perturbation. This is a false positive, not a threat evading detection.
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif status == "robust":
        if actual_class == 1:
            robust_badge = "ATTACK REPELLED"
            robust_title = "AI Remained Robust: Threat Caught Successfully!"
            robust_description = (
                f"The ground-truth label is ATTACK. After the {st.session_state['attack_name']} "
                f"perturbation, the model still correctly classified the flow as ATTACK "
                f"({adversarial_confidence:.1%} confidence)."
            )
        else:
            robust_badge = "BENIGN FLOW"
            robust_title = "Legitimate Traffic Correctly Identified"
            robust_description = (
                f"The ground-truth label is BENIGN. After the {st.session_state['attack_name']} "
                f"perturbation, the model still correctly classified the flow as BENIGN "
                f"({adversarial_confidence:.1%} confidence). No threat was present in this sample."
            )
        st.markdown(
            f"""
            <div style="background: #f0fdf4; border: 1.5px solid #bbf7d0; border-radius: 18px; padding: 1.3rem 1.6rem; margin-bottom: 1.2rem;">
                <div style="display: flex; align-items: flex-start; gap: 1rem;">
                    <div style="font-size: 2rem; background: #dcfce7; border-radius: 14px; padding: 0.5rem 0.75rem; border: 1px solid #86efac;">
                        🛡️
                    </div>
                    <div>
                        <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                            <span class="badge-mint mono">{robust_badge}</span>
                            <span style="font-size: 1.05rem; font-weight: 800; color: #166534;">
                                {robust_title}
                            </span>
                        </div>
                        <p style="font-size: 0.88rem; color: #475569; margin-top: 0.45rem; line-height: 1.6; margin-bottom: 0;">
                            {robust_description}
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="background: #fff7ed; border: 1.5px solid #fed7aa; border-radius: 18px; padding: 1.3rem 1.6rem; margin-bottom: 1.2rem;">
                <div style="display: flex; align-items: flex-start; gap: 1rem;">
                    <div style="font-size: 2rem; background: #ffedd5; border-radius: 14px; padding: 0.5rem 0.75rem; border: 1px solid #fdba74;">
                        ℹ️
                    </div>
                    <div>
                        <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                            <span class="badge-peach mono">BASELINE ANOMALY</span>
                            <span style="font-size: 1.05rem; font-weight: 800; color: #9a3412;">
                                Clean Traffic Baseline Was Already Incorrect
                            </span>
                        </div>
                        <p style="font-size: 0.88rem; color: #475569; margin-top: 0.45rem; line-height: 1.6; margin-bottom: 0;">
                            The ground truth label is <strong>{class_name(actual_class)}</strong>, but the model had already misclassified the clean packet as <strong>{class_name(clean_prediction)}</strong> before any attack was attempted.
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 2. Before vs After Packet Story
    b_col1, b_col2 = st.columns(2)

    with b_col1:
        st.markdown(
            f"""
            <div class="card-white">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.6rem; margin-bottom: 0.9rem;">
                    <span style="font-weight: 800; color: #0f172a; font-size: 0.95rem;">
                        1️⃣ What It Looked Like Initially
                    </span>
                    <span class="badge-mint mono">REAL FLOW</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.8rem;">
                    <div>
                        <div style="font-size: 0.74rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Original AI Verdict</div>
                        <div style="font-size: 1.65rem; font-weight: 800; color: {'#e11d48' if clean_prediction == 1 else '#16a34a'};">
                            {class_name(clean_prediction)}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.74rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Confidence</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #0f172a;" class="mono">
                            {clean_confidence:.2%}
                        </div>
                    </div>
                </div>
                <div style="background: #f1f5f9; border-radius: 9999px; height: 8px; overflow: hidden; margin-bottom: 0.6rem;">
                    <div style="background: {'#f43f5e' if clean_prediction == 1 else '#22c55e'}; width: {clean_confidence*100:.1f}%; height: 100%; border-radius: 9999px;"></div>
                </div>
                <div style="font-size: 0.78rem; color: #64748b;">
                    Ground Truth: <strong>{class_name(actual_class)}</strong> • Model easily identified standard signatures.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with b_col2:
        conf_diff = adversarial_confidence - clean_confidence
        st.markdown(
            f"""
            <div class="card-white">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.6rem; margin-bottom: 0.9rem;">
                    <span style="font-weight: 800; color: #0f172a; font-size: 0.95rem;">
                        2️⃣ After {st.session_state['attack_name']} Perturbation
                    </span>
                    <span class="badge-rose mono">{st.session_state['attack_name']} PERTURBATION</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.8rem;">
                    <div>
                        <div style="font-size: 0.74rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Post-Perturbation Verdict</div>
                        <div style="font-size: 1.65rem; font-weight: 800; color: {'#e11d48' if adversarial_prediction == 1 else '#16a34a'};">
                            {class_name(adversarial_prediction)} {'⚠️' if status == 'success' else ''}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.74rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Confidence Shift</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: {'#dc2626' if conf_diff < 0 else '#16a34a'};" class="mono">
                            {conf_diff:+.2%}
                        </div>
                    </div>
                </div>
                <div style="background: #f1f5f9; border-radius: 9999px; height: 8px; overflow: hidden; margin-bottom: 0.6rem;">
                    <div style="background: {'#f43f5e' if adversarial_prediction == 1 else '#22c55e'}; width: {adversarial_confidence*100:.1f}%; height: 100%; border-radius: 9999px;"></div>
                </div>
                <div style="font-size: 0.78rem; color: #64748b;">
                    Verdict Changed: <strong>{'YES (EVASION)' if status == 'success' and actual_class == 1 else 'YES (FALSE ALARM)' if status == 'success' and actual_class == 0 else 'NO (ROBUST)'}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 3. Show a narrative that matches the sample's ground truth and outcome.
    bx_col1, bx_col2 = st.columns(2)
    current_att = st.session_state["attack_name"]

    if actual_class == 0:
        with bx_col1:
            st.markdown(
                f"""
                <div class="card-white" style="border-left: 4px solid #16a34a; height: 100%;">
                    <div style="font-size: 0.88rem; font-weight: 800; color: #166534; margin-bottom: 0.4rem;">
                        ✅ Benign Flow: No Threat to Explain
                    </div>
                    <p style="font-size: 0.82rem; color: #475569; margin: 0; line-height: 1.6;">
                        Ground truth is BENIGN. The model classified the original flow as
                        {class_name(clean_prediction)} and the perturbed flow as
                        {class_name(adversarial_prediction)}. This sample does not show an attacker evading detection.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with bx_col2:
            st.markdown(
                """
                <div class="card-white" style="border-left: 4px solid #94a3b8; height: 100%;">
                    <div style="font-size: 0.88rem; font-weight: 800; color: #334155; margin-bottom: 0.4rem;">
                        ℹ️ No Threat-Specific Fix Needed
                    </div>
                    <p style="font-size: 0.82rem; color: #475569; margin: 0; line-height: 1.6;">
                        This flow is labeled BENIGN. Review system defenses using confirmed attack samples.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        if status == "success":
            if added_features:
                feat_bullets = "".join([f"<li style='margin-bottom: 0.25rem;'><strong>{f}</strong> changed in the perturbed flow.</li>" for f in added_features])
            else:
                feat_bullets = "<li>The perturbed flow's feature values changed.</li>"
            explanation_title = "How the Perturbation Fooled the AI"
            explanation_text = "The model correctly identified the original attack but classified the perturbed flow as BENIGN."
        elif status == "robust":
            feat_bullets = ""
            explanation_title = "The AI Still Detected the Attack"
            explanation_text = "The ground-truth label is ATTACK, and the model classified both the original and perturbed flow correctly."
        else:
            feat_bullets = ""
            explanation_title = "The Baseline Prediction Was Already Wrong"
            explanation_text = (
                f"Ground truth is ATTACK. The model classified the original flow as "
                f"{class_name(clean_prediction)} and the perturbed flow as "
                f"{class_name(adversarial_prediction)}."
            )
        feat_bullets_html = (
            f"<ul style='font-size: 0.82rem; color: #475569; padding-left: 1.2rem; margin: 0; line-height: 1.6;'>{feat_bullets}</ul>"
            if feat_bullets else ""
        )
        explanation_body_html = (
            f"{feat_bullets_html}<p style='font-size: 0.76rem; color: #64748b; margin-top: 0.6rem; margin-bottom: 0;'>"
            f"{explanation_text}</p>"
        )

        with bx_col1:
            st.markdown(
                f"""
                <div class="card-white" style="border-left: 4px solid #7c3aed; height: 100%;">
                    <div style="font-size: 0.88rem; font-weight: 800; color: #581c87; margin-bottom: 0.4rem;">
                        💡 {explanation_title}:
                    </div>
                    {explanation_body_html}
                </div>
                """,
                unsafe_allow_html=True
            )

        with bx_col2:
            if current_att == "DeepFool":
                rec_text = "Adversarial retraining can help the model recognize DeepFool-style perturbations."
            else:
                rec_text = "Evaluate feature squeezing against this perturbation using the full validation set."
            st.markdown(
                f"""
                <div class="card-white" style="border-left: 4px solid #16a34a; height: 100%;">
                    <div style="font-size: 0.88rem; font-weight: 800; color: #166534; margin-bottom: 0.4rem;">
                        🛡️ Defense to Evaluate
                    </div>
                    <p style="font-size: 0.82rem; color: #475569; margin: 0; line-height: 1.6;">
                        {rec_text}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # 4. Simple Benchmark Impact Callout
    acc_clean = 95.04
    acc_curr = ATTACK_RESULTS.get(st.session_state["attack_name"], {}).get("accuracy", 0.9504) * 100
    st.markdown(
        f"""
        <div class="card-white" style="margin-top: 0.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.6rem;">
                <span style="font-size: 0.88rem; font-weight: 800; color: #0f172a;">
                    📉 Overall Impact on System Accuracy
                </span>
                <span class="badge-peach mono">{st.session_state['attack_name']} Vector</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 0.8rem;">
                <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 0.75rem 1rem;">
                    <div style="font-size: 0.74rem; font-weight: 700; color: #166534;">NORMAL ACCURACY</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #166534;" class="mono">{acc_clean:.1f}%</div>
                </div>
                <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 12px; padding: 0.75rem 1rem;">
                    <div style="font-size: 0.74rem; font-weight: 700; color: #991b1b;">ACCURACY UNDER {st.session_state['attack_name'].upper()}</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #991b1b;" class="mono">{acc_curr:.1f}%</div>
                </div>
            </div>
            <p style="font-size: 0.78rem; color: #64748b; margin: 0;">
                Looking for raw SHAP values, mathematical feature matrices, or full pipeline charts?
                <strong>Switch to Analyst Deep-Dive</strong> in the sidebar or using the button above.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# BRANCH 2: ANALYST DEEP-DIVE (FULL 3-TAB CONSOLE UNLOCKED)
# ============================================================

else:
    tab_telemetry, tab_benchmarks, tab_defense = st.tabs([
        "🖥️ Threat Telemetry & SHAP Vectors",
        "📊 Model Resilience Scorecard",
        "🛡️ Defense Strategy & Hardening Pipeline"
    ])

    with tab_telemetry:
        st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)

        # Full 4-Metric Telemetry Strip
        st.markdown(
            """
            <div style="font-size: 0.85rem; font-weight: 800; color: #334155; margin-bottom: 0.6rem;">
                🔬 Explainable AI (XAI) Behavior Telemetry Strip
            </div>
            """,
            unsafe_allow_html=True
        )

        t1, t2, t3, t4 = st.columns(4)
        with t1:
            st.metric(
                label="Explanation Drift (L1)",
                value=f"{local_drift:.4f}",
                delta="Low Shift" if local_drift < 0.02 else "High Shift"
            )
        with t2:
            st.metric(
                label="SHAP Cosine Similarity",
                value=f"{local_similarity:.1%}",
                delta="Preserved" if local_similarity > 0.6 else "Scrambled"
            )
        with t3:
            st.metric(
                label="Top-10 Feature Overlap",
                value=f"{local_overlap:.0%}",
                delta="Stable Features" if local_overlap >= 0.6 else "High Distortion"
            )
        with t4:
            st.metric(
                label="Vector Severity",
                value=ATTACK_RESULTS.get(st.session_state["attack_name"], {}).get("severity", "Moderate"),
                delta=st.session_state["attack_name"]
            )

        st.markdown('<div style="height: 0.6rem;"></div>', unsafe_allow_html=True)

        # Side-by-Side Model Prediction Coordinates
        an_col1, an_col2 = st.columns(2)

        with an_col1:
            st.markdown(
                f"""
                <div class="card-white">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.5rem; margin-bottom: 0.8rem;">
                        <span style="font-weight: 800; font-size: 0.9rem;">Clean Flow Vector</span>
                        <span class="badge-mint mono">Ground Truth: {class_name(actual_class)}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #64748b; line-height: 1.7;">
                        <div>Predicted Class: <strong>{class_name(clean_prediction)}</strong></div>
                        <div>Output Probability: <span class="mono">p = {clean_probability:.4f}</span></div>
                        <div>Confidence Score: <span class="mono">{clean_confidence:.2%}</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with an_col2:
            conf_diff = adversarial_confidence - clean_confidence
            st.markdown(
                f"""
                <div class="card-white">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.5rem; margin-bottom: 0.8rem;">
                        <span style="font-weight: 800; font-size: 0.9rem;">Adversarial Flow Vector</span>
                        <span class="badge-rose mono">Method: {st.session_state['attack_name']}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #64748b; line-height: 1.7;">
                        <div>Predicted Class: <strong>{class_name(adversarial_prediction)}</strong></div>
                        <div>Output Probability: <span class="mono">p = {adversarial_probability:.4f}</span></div>
                        <div>Confidence Delta: <span class="mono" style="color: {'#dc2626' if conf_diff < 0 else '#16a34a'}; font-weight: 700;">{conf_diff:+.2%}</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Complete 10-Feature SHAP Contribution Matrix (Unlocked)
        st.markdown(
            """
            <div style="font-size: 0.88rem; font-weight: 800; color: #0f172a; margin-top: 0.6rem; margin-bottom: 0.4rem;">
                🔬 10-Feature Contribution Matrix (SHAP Values & Directional Impact)
            </div>
            <div style="font-size: 0.78rem; color: #64748b; margin-bottom: 0.6rem;">
                Positive SHAP pushes model toward <span class="badge-rose" style="padding: 0.1rem 0.4rem;">ATTACK</span>, negative pushes toward <span class="badge-mint" style="padding: 0.1rem 0.4rem;">BENIGN</span>.
            </div>
            """,
            unsafe_allow_html=True
        )

        comparison_indices = np.argsort(
            np.maximum(np.abs(clean_shap), np.abs(adversarial_shap))
        )[::-1][:10]

        comparison_rows = []
        for idx in comparison_indices:
            clean_v = float(clean_shap[idx])
            adv_v = float(adversarial_shap[idx])
            comparison_rows.append({
                "Feature Name": feature_names[idx],
                "Clean SHAP": clean_v,
                "Adversarial SHAP": adv_v,
                "Delta (Shift)": adv_v - clean_v,
                "Attacker Effect": "Shifted to Benign 🟢" if (adv_v - clean_v) < 0 else "Shifted to Attack 🔴"
            })

        comp_df = pd.DataFrame(comparison_rows)

        def style_clean_adv(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return "background-color: #FEE2E2; color: #991B1B; font-weight: 700;"
                elif val < 0:
                    return "background-color: #DCFCE7; color: #166534; font-weight: 700;"
            return ""

        light_table_styles = [
            {"selector": "th", "props": [("background-color", "#f8fafc"), ("color", "#334155"), ("border", "1px solid #e2e8f0")]},
            {"selector": "td", "props": [("background-color", "#ffffff"), ("color", "#334155"), ("border", "1px solid #e2e8f0")]},
        ]
        styled_df = (
            comp_df.style
            .format({
                "Clean SHAP": "{:+.4f}",
                "Adversarial SHAP": "{:+.4f}",
                "Delta (Shift)": "{:+.4f}"
            })
            .map(style_clean_adv, subset=["Clean SHAP", "Adversarial SHAP"])
            .set_table_styles(light_table_styles)
        )

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

    with tab_benchmarks:
        st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-bottom: 0.3rem;">
                Global Attack Resilience Scorecard
            </div>
            <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 1rem;">
                Evaluating model defense limits across 45,000 independent network flows in the CICIDS2017 benchmark.
            </div>
            """,
            unsafe_allow_html=True
        )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Clean Test Accuracy", "95.04%", "Baseline Target")
        with m2:
            st.metric("FGSM Accuracy", "88.78%", "-6.26% Drop", delta_color="inverse")
        with m3:
            st.metric("PGD Accuracy", "87.83%", "-7.21% Drop", delta_color="inverse")
        with m4:
            st.metric("DeepFool Accuracy", "5.37%", "-89.67% CRITICAL", delta_color="inverse")

        st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)

        chart_c1, chart_c2 = st.columns([1.1, 1.4])

        with chart_c1:
            st.markdown("**Accuracy Collapse Under Attacks**")
            acc_df = pd.DataFrame({
                "Scenario": ["Clean Flow", "FGSM", "PGD", "DeepFool"],
                "Accuracy": [95.04, 88.78, 87.83, 5.37],
            })
            fig_acc = px.bar(
                acc_df,
                x="Scenario",
                y="Accuracy",
                text="Accuracy",
                color="Scenario",
                color_discrete_map={
                    "Clean Flow": "#86efac",
                    "FGSM": "#fed7aa",
                    "PGD": "#fdba74",
                    "DeepFool": "#fca5a5"
                }
            )
            fig_acc.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
                marker_line_color="#94a3b8",
                marker_line_width=1
            )
            fig_acc.update_layout(
                height=320,
                showlegend=False,
                yaxis_range=[0, 110],
                yaxis_title="Accuracy (%)",
                xaxis_title="",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=15, r=15, t=15, b=15)
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with chart_c2:
            st.markdown("**Top Vulnerable Features Disrupted by Attackers**")
            top_feat_clean = global_importance.sort_values("Clean", ascending=False).head(7)
            fig_feat = go.Figure()
            fig_feat.add_trace(go.Bar(
                y=top_feat_clean["Feature"],
                x=top_feat_clean["Clean"],
                name="Clean Flow",
                orientation="h",
                marker=dict(color="#a7f3d0", line=dict(color="#10b981", width=1))
            ))
            fig_feat.add_trace(go.Bar(
                y=top_feat_clean["Feature"],
                x=top_feat_clean["DeepFool"],
                name="Under DeepFool",
                orientation="h",
                marker=dict(color="#fecaca", line=dict(color="#ef4444", width=1))
            ))
            fig_feat.update_layout(
                height=320,
                barmode="group",
                yaxis=dict(autorange="reversed"),
                xaxis_title="Mean |SHAP Value|",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=15, r=15, t=15, b=15)
            )
            st.plotly_chart(fig_feat, use_container_width=True)

        # Full Benchmark Matrix Table
        st.markdown("**Complete Attack Telemetry Matrix**")
        bench_df = pd.DataFrame([
            {
                "Attack Vector": "FGSM (Fast Gradient Sign)",
                "Mechanism": "Single-step L-infinity bound",
                "Accuracy": "88.78%",
                "Explanation Drift": "0.0159",
                "SHAP Similarity": "67.96%",
                "Top-10 Overlap": "67.74%",
                "Threat Level": "Moderate ⚠️"
            },
            {
                "Attack Vector": "PGD (Projected Gradient Descent)",
                "Mechanism": "Iterative multi-step projection",
                "Accuracy": "87.83%",
                "Explanation Drift": "0.0162",
                "SHAP Similarity": "66.45%",
                "Top-10 Overlap": "67.76%",
                "Threat Level": "Moderate ⚠️"
            },
            {
                "Attack Vector": "DeepFool",
                "Mechanism": "Hyperplane boundary push",
                "Accuracy": "5.37%",
                "Explanation Drift": "0.0278",
                "SHAP Similarity": "36.02%",
                "Top-10 Overlap": "44.94%",
                "Threat Level": "CRITICAL 🚨"
            }
        ])
        st.dataframe(
            bench_df.style.set_table_styles(light_table_styles),
            use_container_width=True,
            hide_index=True
        )

    with tab_defense:
        st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)

        current_att = st.session_state["attack_name"]
        if current_att == "DeepFool":
            rec_pill = "badge-rose"
            rec_title = "Adversarial Retraining + Input Quantization"
            rec_desc = "DeepFool's minimal boundary perturbation causes extreme evasion (-89.67% accuracy drop). Retraining the LSTM directly on synthetic DeepFool samples is the primary effective defense."
        else:
            rec_pill = "badge-peach"
            rec_title = "Feature Squeezing + Spatial Smoothing"
            rec_desc = f"{current_att} relies on small uniform gradient noise. Feature squeezing reduces input bit-depth to eliminate high-frequency adversarial noise without hurting clean traffic throughput."

        st.markdown(
            f"""
            <div class="card-white" style="border-left: 5px solid #7c3aed; margin-bottom: 1.2rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                    <span class="{rec_pill}">AUTOMATED RECOMMENDATION</span>
                    <span class="badge-lavender mono">VECTOR: {current_att}</span>
                </div>
                <h3 style="margin: 0.6rem 0 0.3rem 0; font-size: 1.15rem; font-weight: 800; color: #0f172a;">
                    🛡️ Recommended Countermeasure: {rec_title}
                </h3>
                <p style="margin: 0; font-size: 0.85rem; color: #475569; line-height: 1.55;">
                    {rec_desc}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        d1, d2, d3 = st.columns(3)

        with d1:
            st.markdown(
                """
                <div class="card-white">
                    <span class="badge-mint">STRATEGY 1</span>
                    <h4 style="margin: 0.6rem 0 0.2rem 0; font-size: 0.95rem; font-weight: 800; color: #0f172a;">
                        Feature Squeezing
                    </h4>
                    <p style="font-size: 0.78rem; color: #64748b; line-height: 1.5; margin-bottom: 0.6rem;">
                        Reduces input bit-depth resolution and applies spatial smoothing to strip out tiny adversarial noise.
                    </p>
                    <div style="font-size: 0.74rem; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 0.4rem 0.6rem; color: #166534; font-weight: 700;">
                        Best for: FGSM & PGD gradient attacks
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with d2:
            st.markdown(
                """
                <div class="card-white">
                    <span class="badge-rose">STRATEGY 2</span>
                    <h4 style="margin: 0.6rem 0 0.2rem 0; font-size: 0.95rem; font-weight: 800; color: #0f172a;">
                        Adversarial Retraining
                    </h4>
                    <p style="font-size: 0.78rem; color: #64748b; line-height: 1.5; margin-bottom: 0.6rem;">
                        Injects synthetic adversarial vectors directly into the neural network's training minibatches.
                    </p>
                    <div style="font-size: 0.74rem; background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 0.4rem 0.6rem; color: #991b1b; font-weight: 700;">
                        Required for: DeepFool boundary evasion
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with d3:
            st.markdown(
                """
                <div class="card-white">
                    <span class="badge-lavender">STRATEGY 3</span>
                    <h4 style="margin: 0.6rem 0 0.2rem 0; font-size: 0.95rem; font-weight: 800; color: #0f172a;">
                        Randomized Smoothing
                    </h4>
                    <p style="font-size: 0.78rem; color: #64748b; line-height: 1.5; margin-bottom: 0.6rem;">
                        Adds a Gaussian noise envelope around test samples to establish provable mathematical security bounds.
                    </p>
                    <div style="font-size: 0.74rem; background: #faf5ff; border: 1px solid #e9d5ff; border-radius: 8px; padding: 0.4rem 0.6rem; color: #6b21a8; font-weight: 700;">
                        Best for: Certified robust guarantees
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ============================================================
# CONSOLE FOOTER
# ============================================================

st.markdown(
    """
    <div style="text-align: center; margin-top: 2rem; padding: 1.2rem 0; border-top: 1px solid #e2e8f0; font-size: 0.76rem; color: #94a3b8;">
        🛡️ SentinelXAI Console • Explainable AI Security Defense Center • Built with TensorFlow, SHAP & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
