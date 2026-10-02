import json

from pathlib import Path



import numpy as np

import pandas as pd

import streamlit as st

import tensorflow as tf

import plotly.express as px





# ============================================================

# PAGE CONFIGURATION

# ============================================================



st.set_page_config(

    page_title="SentinelXAI",

    page_icon="🛡️",

    layout="wide",

    initial_sidebar_state="expanded"

)





# ============================================================

# CUSTOM CSS

# ============================================================



st.markdown(

    """

    <style>



        .main-title {

            font-size: 2.4rem;

            font-weight: 700;

            margin-bottom: 0.1rem;

        }



        .subtitle {

            font-size: 1.05rem;

            color: #b0b0b0;

            margin-bottom: 1.5rem;

        }



        .section-title {

            font-size: 1.55rem;

            font-weight: 650;

            margin-top: 1.4rem;

            margin-bottom: 0.6rem;

        }



        .info-box {

            padding: 1.15rem 1.3rem;

            border-radius: 10px;

            background-color: #1f232b;

            color: #f0f0f0;

            border: 1px solid #3a404c;

            margin: 0.8rem 0;

            line-height: 1.6;

        }



        .small-note {

            font-size: 0.85rem;

            color: #aaaaaa;

        }



    </style>

    """,

    unsafe_allow_html=True

)





# ============================================================

# PROJECT PATHS

# ============================================================



PROJECT_DIR = Path(r"C:\SentinelXAI")



MODEL_PATH = PROJECT_DIR / "models" / "baseline_lstm.keras"



DATA_DIR = PROJECT_DIR / "data" / "processed"

RESULTS_DIR = PROJECT_DIR / "results"



X_TEST_PATH = DATA_DIR / "X_test_lstm.npy"

Y_TEST_PATH = DATA_DIR / "y_test_lstm.npy"



# Exact filenames from the previous working dashboard

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

# LOAD MODEL

# ============================================================



@st.cache_resource

def load_model():



    return tf.keras.models.load_model(

        MODEL_PATH

    )





# ============================================================

# LOAD DATA

# ============================================================



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



    behavior = pd.read_csv(

        BEHAVIOR_PATH

    )



    with open(

        FEATURE_NAMES_PATH,

        "r",

        encoding="utf-8"

    ) as f:



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





# ============================================================

# LOAD PROJECT

# ============================================================



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



    st.error(

        "Unable to load SentinelXAI project files."

    )



    st.exception(e)



    st.stop()





# ============================================================

# ATTACK RESULTS

# ============================================================



ATTACK_RESULTS = {



    "FGSM": {

        "accuracy": 0.8878,

        "drift": 0.0158611501,

        "similarity": 0.67960357,

        "overlap": 0.6774,

        "confidence_drop": 0.023536352

    },



    "PGD": {

        "accuracy": 0.8783,

        "drift": 0.0162214949,

        "similarity": 0.66450081,

        "overlap": 0.6776,

        "confidence_drop": 0.026820794

    },



    "DeepFool": {

        "accuracy": 0.0537,

        "drift": 0.0277837812,

        "similarity": 0.36024411,

        "overlap": 0.4494,

        "confidence_drop": 0.6168139

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



    if int(label) == 1:



        return "ATTACK"



    return "BENIGN"





def get_prediction(sample):



    probability = float(

        model.predict(

            sample,

            verbose=0

        )[0][0]

    )



    prediction = (

        1

        if probability >= 0.5

        else 0

    )



    if prediction == 1:



        confidence = probability



    else:



        confidence = 1 - probability



    return (

        prediction,

        confidence,

        probability

    )





def get_status(

    clean_prediction,

    adversarial_prediction,

    actual_class

):



    clean_correct = (

        clean_prediction == actual_class

    )



    adversarial_correct = (

        adversarial_prediction == actual_class

    )



    if clean_correct and not adversarial_correct:



        return "success"



    if clean_correct and adversarial_correct:



        return "robust"



    return "clean_wrong"





def calculate_sample_metrics(

    clean_shap,

    adversarial_shap

):



    drift = float(

        np.mean(

            np.abs(

                clean_shap - adversarial_shap

            )

        )

    )



    clean_norm = np.linalg.norm(

        clean_shap

    )



    adversarial_norm = np.linalg.norm(

        adversarial_shap

    )



    if (

        clean_norm == 0

        or adversarial_norm == 0

    ):



        similarity = 0.0



    else:



        similarity = float(

            np.dot(

                clean_shap,

                adversarial_shap

            )

            /

            (

                clean_norm

                * adversarial_norm

            )

        )



    clean_top10 = set(

        np.argsort(

            np.abs(clean_shap)

        )[-10:]

    )



    adversarial_top10 = set(

        np.argsort(

            np.abs(adversarial_shap)

        )[-10:]

    )



    overlap = (

        len(

            clean_top10.intersection(

                adversarial_top10

            )

        )

        / 10

    )



    return (

        drift,

        similarity,

        overlap

    )





def explain_prediction(

    shap_values,

    prediction,

    names,

    n=3

):



    shap_values = np.asarray(shap_values)



    top_indices = np.argsort(

        np.abs(shap_values)

    )[::-1][:n]



    direction = class_name(prediction)



    lines = [



        f"**Model prediction: {direction}**",



        "",



        "**Strongest individual feature contributions:**"



    ]



    for idx in top_indices:



        value = float(

            shap_values[idx]

        )



        feature = names[idx]



        if value > 0:



            direction_text = "toward ATTACK"



        elif value < 0:



            direction_text = "toward BENIGN"



        else:



            direction_text = "very little directional influence"



        lines.append(



            f"- **{feature}**: `{value:.4f}` → "

            f"{direction_text}"



        )



    lines.extend([

        "",

        "ℹ️ **Note:** The final prediction is based on the "

        "**combined effect of all 78 features**, not just these "

        "top contributions."

    ])



    return "\n".join(lines)



def generate_change_description(
    attack_name,
    clean_prediction,
    adversarial_prediction,
    actual_class,
    clean_confidence,
    adversarial_confidence,
    clean_shap,
    adversarial_shap,
    feature_names
):

    status = get_status(
        clean_prediction,
        adversarial_prediction,
        actual_class
    )

    drift, similarity, overlap = calculate_sample_metrics(
        clean_shap,
        adversarial_shap
    )

    # Top-5 features are used only to describe what changed.
    clean_top_indices = np.argsort(
        np.abs(clean_shap)
    )[::-1][:5]

    adversarial_top_indices = np.argsort(
        np.abs(adversarial_shap)
    )[::-1][:5]

    clean_top_set = set(clean_top_indices)
    adversarial_top_set = set(adversarial_top_indices)

    added_features = [
        feature_names[idx]
        for idx in adversarial_top_indices
        if idx not in clean_top_set
    ][:2]

    removed_features = [
        feature_names[idx]
        for idx in clean_top_indices
        if idx not in adversarial_top_set
    ][:2]

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if status == "success":

        status_text = (
            f"🔴 **Model was fooled by {attack_name}:** "
            f"{class_name(clean_prediction)} → "
            f"{class_name(adversarial_prediction)}."
        )

    elif status == "robust":

        status_text = (
            f"🟢 **Model remained robust to {attack_name}:** "
            f"both predictions were **{class_name(actual_class)}**."
        )

    else:

        status_text = (
            "🟡 **Clean prediction was already incorrect;** "
            "evasion success is not counted."
        )

    # --------------------------------------------------------
    # FEATURE CHANGES
    # --------------------------------------------------------

    feature_parts = []

    if added_features:

        feature_parts.append(
            f"**New prominent:** {', '.join(added_features)}."
        )

    if removed_features:

        feature_parts.append(
            f"**Less prominent:** {', '.join(removed_features)}."
        )

    if feature_parts:

        feature_text = " ".join(feature_parts)

    else:

        feature_text = (
            "**Main features remained relatively similar.**"
        )

    # --------------------------------------------------------
    # FINAL SHORT DESCRIPTION
    # --------------------------------------------------------

    return f"""
{status_text}

### Confidence: **{clean_confidence:.2%} → {adversarial_confidence:.2%}**

**Explanation Drift:** `{drift:.4f}`

{feature_text}

**SHAP comparison** shows how feature contributions changed after the attack.
"""


# ============================================================

# GLOBAL SHAP IMPORTANCE

# ============================================================



@st.cache_data

def calculate_global_importance(

    clean_shap,

    fgsm_shap,

    pgd_shap,

    deepfool_shap,

    feature_names

):



    clean_importance = np.mean(

        np.abs(clean_shap),

        axis=0

    )



    fgsm_importance = np.mean(

        np.abs(fgsm_shap),

        axis=0

    )



    pgd_importance = np.mean(

        np.abs(pgd_shap),

        axis=0

    )



    deepfool_importance = np.mean(

        np.abs(deepfool_shap),

        axis=0

    )



    return pd.DataFrame(

        {

            "Feature": feature_names,



            "Clean": clean_importance,



            "FGSM": fgsm_importance,



            "PGD": pgd_importance,



            "DeepFool": deepfool_importance

        }

    )





global_importance = calculate_global_importance(

    shap_clean,

    shap_fgsm,

    shap_pgd,

    shap_deepfool,

    feature_names

)





# ============================================================

# HEADER

# ============================================================



st.markdown(

    '<div class="main-title">🛡️ SentinelXAI</div>',

    unsafe_allow_html=True

)



st.markdown(

    '<div class="subtitle">'

    "An Explainability-Guided Defense Decision Framework for "

    "Adversarially Robust Intrusion Detection Systems"

    "</div>",

    unsafe_allow_html=True

)





# ============================================================

# SIDEBAR

# ============================================================



st.sidebar.title(

    "SentinelXAI Controls"

)



st.sidebar.markdown(

    """

    **Select a SHAP sample and adversarial attack**



    to investigate how adversarial perturbations affect the IDS

    prediction and its explanations.

    """

)



st.sidebar.divider()





sample_number = st.sidebar.number_input(

    "Select SHAP Sample",

    min_value=0,

    max_value=len(behavior) - 1,

    value=0,

    step=1

)



attack_name = st.sidebar.selectbox(

    "Select Adversarial Attack",

    [

        "FGSM",

        "PGD",

        "DeepFool"

    ]

)



st.sidebar.divider()



st.sidebar.markdown(

    """

    **Sample note**



    The dashboard contains 500 matched SHAP samples selected

    from the 45,000-sample test set.



    Each selected SHAP sample corresponds to one original

    test-set flow.

    """

)





# ============================================================

# SAMPLE MAPPING

# ============================================================



shap_row = int(

    sample_number

)



original_index = int(

    behavior.iloc[

        shap_row

    ]["sample_index"]

)



clean_sample = X_test[

    original_index:

    original_index + 1

]



adversarial_sample = ATTACK_ARRAYS[

    attack_name

][

    original_index:

    original_index + 1

]



clean_shap = shap_clean[

    shap_row

]



adversarial_shap = ATTACK_SHAP[

    attack_name

][

    shap_row

]



actual_class = int(

    y_test[

        original_index

    ]

)





# ============================================================

# MODEL PREDICTIONS

# ============================================================



(

    clean_prediction,

    clean_confidence,

    clean_probability

) = get_prediction(

    clean_sample

)



(

    adversarial_prediction,

    adversarial_confidence,

    adversarial_probability

) = get_prediction(

    adversarial_sample

)



status = get_status(

    clean_prediction,

    adversarial_prediction,

    actual_class

)



# ============================================================

# DATASET & IDS SNAPSHOT

# ============================================================



st.markdown(

    '<div class="section-title">'

    "🔎 IDS & Dataset Snapshot"

    "</div>",

    unsafe_allow_html=True

)



# ------------------------------------------------------------

# QUICK STATS

# ------------------------------------------------------------



d1, d2, d3, d4, d5, d6 = st.columns(6)



with d1:

    st.metric(

        "Original Flows",

        "2.83M"

    )



with d2:

    st.metric(

        "Dataset Used",

        "300K"

    )



with d3:

    st.metric(

        "Training",

        "210K"

    )



with d4:

    st.metric(

        "Validation",

        "45K"

    )



with d5:

    st.metric(

        "Testing",

        "45K"

    )



with d6:

    st.metric(

        "SHAP Samples",

        "500"

    )



# ------------------------------------------------------------

# DATA REDUCTION VISUAL

# ------------------------------------------------------------



st.markdown("### 📦 Data Selection")



selection_col1, selection_col2, selection_col3 = st.columns(

    [1.2, 0.8, 1.2]

)



with selection_col1:



    st.metric(

        "CICIDS2017",

        "2.83M flows"

    )



with selection_col2:



    st.markdown(

        "<div style='text-align:center; "

        "font-size:2rem; padding-top:0.4rem;'>"

        "↓"

        "</div>",

        unsafe_allow_html=True

    )



with selection_col3:



    st.metric(

        "Project Dataset",

        "300K flows"

    )





# ------------------------------------------------------------

# TRAIN / VALIDATION / TEST VISUAL

# ------------------------------------------------------------



st.markdown("### 🧪 Model Data Split")



split_data = pd.DataFrame(

    {

        "Set": [

            "Training",

            "Validation",

            "Testing"

        ],

        "Samples": [

            210000,

            45000,

            45000

        ]

    }

)



fig_split = px.bar(

    split_data,

    x="Set",

    y="Samples",

    text="Samples",

    title="300K Balanced Development Dataset"

)



fig_split.update_traces(

    texttemplate="%{text:,}",

    textposition="outside"

)



fig_split.update_layout(

    height=360,

    yaxis_title="Number of Flows",

    xaxis_title="",

    showlegend=False,

    margin=dict(

        l=20,

        r=20,

        t=60,

        b=20

    )

)



st.plotly_chart(

    fig_split,

    use_container_width=True

)





# ------------------------------------------------------------

# DATASET NOTE

# ------------------------------------------------------------



st.caption(

    "CICIDS2017 → 300K balanced flows → 70/15/15 split → "

    "45K held-out test flows → 500 matched SHAP samples."

)



# ============================================================

# PROJECT OVERVIEW

# ============================================================



st.markdown(

    '<div class="section-title">'

    "Project Overview"

    "</div>",

    unsafe_allow_html=True

)



col1, col2, col3, col4 = st.columns(4)



with col1:



    st.metric(

        "Clean Test Accuracy",

        "95.04%"

    )



with col2:



    st.metric(

        "Adversarial Attacks",

        "3"

    )



with col3:



    st.metric(

        "Input Features",

        "78"

    )



with col4:



    st.metric(

        "SHAP Samples",

        "500"

    )



st.info(

    f"**Current sample:** SHAP sample {shap_row} "

    f"→ original test-set index {original_index}  \n"

    f"**Selected attack:** {attack_name}"

)



st.divider()

# ============================================================

# SAMPLE CLASSIFICATION

# ============================================================



st.markdown(

    '<div class="section-title">'

    "Sample Classification"

    "</div>",

    unsafe_allow_html=True

)



c1, c2, c3 = st.columns(3)



with c1:



    st.metric(

        "Actual Class",

        class_name(

            actual_class

        )

    )



with c2:



    st.metric(

        "Clean Prediction",

        class_name(

            clean_prediction

        ),

        delta=(

            f"{clean_confidence:.2%} confidence"

        )

    )



with c3:



    st.metric(

        "Adversarial Prediction",

        class_name(

            adversarial_prediction

        ),

        delta=(

            f"{adversarial_confidence:.2%} confidence"

        )

    )





# ============================================================
# ADVERSARIAL ATTACK STATUS
# ============================================================

st.markdown(
    '<div class="section-title">'
    "Adversarial Attack Status"
    "</div>",
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# STATUS CARD
# ------------------------------------------------------------

if status == "success":

    st.error(
        f"""
### 🔴 Successful Evasion

**{class_name(clean_prediction)} → {class_name(adversarial_prediction)}**

The **{attack_name}** attack fooled the model.

**Confidence:** {clean_confidence:.2%} → {adversarial_confidence:.2%}
"""
    )

elif status == "robust":

    st.success(
        f"""
### 🟢 Prediction Remained Correct

**{class_name(clean_prediction)} → {class_name(adversarial_prediction)}**

The **{attack_name}** attack did not fool the model.

**Confidence:** {clean_confidence:.2%} → {adversarial_confidence:.2%}
"""
    )

else:

    st.warning(
        f"""
### 🟡 Clean Prediction Incorrect

**Actual:** {class_name(actual_class)}

The clean prediction was already incorrect, so this sample
is not counted as a successful evasion.
"""
    )


# ============================================================
# GLOBAL SHAP FEATURE IMPORTANCE
# TOP 10 ONLY
# ============================================================

st.markdown(
    '<div class="section-title">'
    "Global Feature Importance — SHAP"
    "</div>",
    unsafe_allow_html=True
)

st.caption(
    "🔬 Top 10 features by mean |SHAP| across 500 matched samples."
)



# ============================================================

# CLEAN TOP 10 — VERTICAL BAR CHART

# ============================================================



clean_global = (

    global_importance

    .sort_values(

        "Clean",

        ascending=False

    )

    .head(10)

)



fig_clean = px.bar(

    clean_global,

    x="Feature",

    y="Clean",

    title="Top 10 Features — Clean Samples",

    labels={

        "Clean": "Mean |SHAP value|",

        "Feature": "Feature"

    }

)



fig_clean.update_layout(

    height=600,

    xaxis_tickangle=-45,

    margin=dict(

        l=20,

        r=20,

        t=60,

        b=170

    )

)



st.plotly_chart(

    fig_clean,

    use_container_width=True

)





# ============================================================

# ATTACK TOP 10 — VERTICAL BAR CHART

# ============================================================



attack_global = (

    global_importance

    .sort_values(

        attack_name,

        ascending=False

    )

    .head(10)

)



fig_attack = px.bar(

    attack_global,

    x="Feature",

    y=attack_name,

    title=(

        f"Top 10 Features — "

        f"{attack_name} Samples"

    ),

    labels={

        attack_name: "Mean |SHAP value|",

        "Feature": "Feature"

    }

)



fig_attack.update_layout(

    height=600,

    xaxis_tickangle=-45,

    margin=dict(

        l=20,

        r=20,

        t=60,

        b=170

    )

)



st.plotly_chart(

    fig_attack,

    use_container_width=True

)



st.caption(

    "These are global SHAP importance summaries across the "

    "500-sample explainability subset. They describe average "

    "feature influence, not the selected sample alone."

)


st.divider()


# ============================================================

# EXPLAINABLE AI — SHAP

# ============================================================



st.markdown(

    '<div class="section-title">'

    "Explainable AI — SHAP Analysis"

    "</div>",

    unsafe_allow_html=True

)



st.info(

    """

### 🔬 SHAP



SHAP shows **which features influence each prediction**.



🔴 **Positive SHAP** → pushes toward **ATTACK**  

🟢 **Negative SHAP** → pushes toward **BENIGN**  

📊 **Larger |SHAP|** → stronger influence



**SentinelXAI compares clean vs adversarial SHAP values**

to measure how the model's decision pattern changes.

"""

)



# ============================================================

# WHY DID THE MODEL MAKE THIS PREDICTION?

# ============================================================



st.markdown(

    '<div class="section-title">'

    "Why Did the Model Make This Prediction?"

    "</div>",

    unsafe_allow_html=True

)



st.markdown(

    explain_prediction(

        clean_shap,

        clean_prediction,

        feature_names

    )

)



# ============================================================

# CLEAN VS ADVERSARIAL COMPARISON

# COLORED SHAP TABLE

# ============================================================



st.markdown(

    '<div class="section-title">'

    "Clean vs Adversarial Feature Contributions"

    "</div>",

    unsafe_allow_html=True

)



comparison_indices = np.argsort(

    np.maximum(

        np.abs(clean_shap),

        np.abs(adversarial_shap)

    )

)[::-1][:10]



comparison_rows = []



for idx in comparison_indices:



    clean_value = float(

        clean_shap[idx]

    )



    adversarial_value = float(

        adversarial_shap[idx]

    )



    comparison_rows.append(

        {

            "Feature": feature_names[idx],



            "Clean SHAP": clean_value,



            "Adversarial SHAP": adversarial_value,



            "Change": (

                adversarial_value

                - clean_value

            )

        }

    )



comparison_df = pd.DataFrame(

    comparison_rows

)





# ------------------------------------------------------------

# SHAP COLORING

# ------------------------------------------------------------



def color_shap(value):



    if pd.isna(value):

        return ""



    if value > 0:

        # Red = pushes toward ATTACK

        return (

            "background-color: rgba(255, 70, 70, 0.25); "

            "color: #ff6b6b; "

            "font-weight: 600;"

        )



    elif value < 0:

        # Green = pushes toward BENIGN

        return (

            "background-color: rgba(50, 205, 100, 0.20); "

            "color: #5ee681; "

            "font-weight: 600;"

        )



    return ""





def color_change(value):



    if pd.isna(value):

        return ""



    if value > 0:

        # Positive change = movement toward ATTACK

        return (

            "background-color: rgba(255, 70, 70, 0.20); "

            "color: #ff6b6b; "

            "font-weight: 600;"

        )



    elif value < 0:

        # Negative change = movement toward BENIGN

        return (

            "background-color: rgba(50, 205, 100, 0.18); "

            "color: #5ee681; "

            "font-weight: 600;"

        )



    return ""





styled_comparison = (

    comparison_df

    .style

    .format(

        {

            "Clean SHAP": "{:.4f}",

            "Adversarial SHAP": "{:.4f}",

            "Change": "{:.4f}"

        }

    )

    .map(

        color_shap,

        subset=["Clean SHAP", "Adversarial SHAP"]

    )

    .map(

        color_change,

        subset=["Change"]

    )

)





st.dataframe(

    styled_comparison,

    use_container_width=True,

    hide_index=True

)





# ------------------------------------------------------------

# LEGEND

# ------------------------------------------------------------



st.caption(

    "🔴 Red = contribution toward ATTACK   |   "

    "🟢 Green = contribution toward BENIGN   |   "

    "For Change, red indicates movement toward ATTACK and "

    "green indicates movement toward BENIGN."

)




# ============================================================
# WHAT CHANGED AFTER THE ATTACK?
# ============================================================

st.markdown(
    '<div class="section-title">'
    "What Changed After the Attack?"
    "</div>",
    unsafe_allow_html=True
)

# ============================================================
# ATTACK RESULT CARD
# ============================================================

if status == "success":

    st.error(
        f"""
### 🔴 Successful Evasion

**Prediction:** {class_name(clean_prediction)}
→ **{class_name(adversarial_prediction)}**

The **{attack_name}** attack fooled the model.
"""
    )

elif status == "robust":

    st.success(
        f"""
### 🟢 Model Remained Robust

**Prediction:** {class_name(clean_prediction)}
→ **{class_name(adversarial_prediction)}**

The **{attack_name}** attack did not change the prediction.
"""
    )

else:

    st.warning(
        f"""
### 🟡 Clean Prediction Incorrect

**Actual:** {class_name(actual_class)}

This sample is not counted as successful evasion.
"""
    )


# ============================================================
# EXPLANATION CHANGE
# ============================================================

st.markdown(
    '<div class="section-title">'
    "🔬 Explanation Change"
    "</div>",
    unsafe_allow_html=True
)

(
    local_drift,
    local_similarity,
    local_overlap
) = calculate_sample_metrics(
    clean_shap,
    adversarial_shap
)

m1, m2, m3 = st.columns(3)

with m1:

    st.metric(
        "Explanation Drift",
        f"{local_drift:.4f}"
    )

    st.caption(
        "Change in SHAP explanation"
    )

with m2:

    st.metric(
        "SHAP Similarity",
        f"{local_similarity:.4f}"
    )

    st.caption(
        "Similarity of explanations"
    )

with m3:

    st.metric(
        "Top-10 Overlap",
        f"{local_overlap:.0%}"
    )

    st.caption(
        "Shared important features"
    )


# ============================================================
# FEATURE IMPORTANCE CHANGES
# ============================================================

st.markdown(
    '<div class="section-title">'
    "📌 Feature Importance Changes"
    "</div>",
    unsafe_allow_html=True
)

clean_top = set(
    np.argsort(
        np.abs(clean_shap)
    )[-5:]
)

attack_top = set(
    np.argsort(
        np.abs(adversarial_shap)
    )[-5:]
)

added_features = [
    feature_names[i]
    for i in attack_top - clean_top
][:3]

removed_features = [
    feature_names[i]
    for i in clean_top - attack_top
][:3]


f1, f2 = st.columns(2)

with f1:

    st.markdown(
        "#### 🆕 More Prominent"
    )

    if added_features:

        for feature in added_features:

            st.markdown(
                f"• **{feature}**"
            )

    else:

        st.caption(
            "No major new features."
        )


with f2:

    st.markdown(
        "#### 📉 Less Prominent"
    )

    if removed_features:

        for feature in removed_features:

            st.markdown(
                f"• **{feature}**"
            )

    else:

        st.caption(
            "No major features reduced."
        )


# ============================================================
# PREDICTION CONFIDENCE
# ============================================================

st.markdown(
    '<div class="section-title">'
    "🎯 Prediction Confidence"
    "</div>",
    unsafe_allow_html=True
)

confidence_change = (
    adversarial_confidence
    - clean_confidence
)

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Clean",
        f"{clean_confidence:.2%}"
    )

    st.caption(
        "Before attack"
    )

with c2:

    st.metric(
        "Adversarial",
        f"{adversarial_confidence:.2%}"
    )

    st.caption(
        "After attack"
    )

with c3:

    st.metric(
        "Change",
        f"{confidence_change:+.2%}"
    )

    st.caption(
        "Confidence difference"
    )


# ============================================================
# QUICK TAKEAWAY
# ============================================================

if status == "success":

    st.error(
        f"🔴 **Takeaway:** {attack_name} changed the model's "
        "prediction for this sample."
    )

elif status == "robust":

    st.success(
        f"🟢 **Takeaway:** {attack_name} changed the model's "
        "explanation, but not its final prediction."
    )

else:

    st.warning(
        "🟡 **Takeaway:** The clean prediction was already incorrect."
    )


# ============================================================
# AGGREGATE XAI BEHAVIOR
# ============================================================

st.markdown(
    '<div class="section-title">'
    "Aggregate XAI Behavior Analysis"
    "</div>",
    unsafe_allow_html=True
)

st.caption(
    "500 matched SHAP samples"
)

result = ATTACK_RESULTS[
    attack_name
]

a1, a2, a3, a4 = st.columns(4)

with a1:
    st.metric(
        "Explanation Drift",
        f"{result['drift']:.4f}"
    )
    st.caption("Change in SHAP explanation")

with a2:
    st.metric(
        "SHAP Similarity",
        f"{result['similarity']:.4f}"
    )
    st.caption("Similarity of explanations")

with a3:
    st.metric(
        "Top-10 Overlap",
        f"{result['overlap']:.1%}"
    )
    st.caption("Shared important features")

with a4:
    st.metric(
        "Confidence Drop",
        f"{result['confidence_drop']:.2%}"
    )
    st.caption("Change in model confidence")


# ============================================================
# ATTACK COMPARISON
# ============================================================

st.markdown(
    '<div class="section-title">'
    "Adversarial Attack Comparison"
    "</div>",
    unsafe_allow_html=True
)

attack_comparison = pd.DataFrame(
    [
        {
            "Attack": "FGSM",
            "Adversarial Accuracy": "88.78%",
            "Explanation Drift": "0.0159",
            "SHAP Similarity": "0.6796",
            "Top-10 Overlap": "67.74%",
            "Confidence Drop": "2.35%"
        },

        {
            "Attack": "PGD",
            "Adversarial Accuracy": "87.83%",
            "Explanation Drift": "0.0162",
            "SHAP Similarity": "0.6645",
            "Top-10 Overlap": "67.76%",
            "Confidence Drop": "2.68%"
        },

        {
            "Attack": "DeepFool",
            "Adversarial Accuracy": "5.37%",
            "Explanation Drift": "0.0278",
            "SHAP Similarity": "0.3602",
            "Top-10 Overlap": "44.94%",
            "Confidence Drop": "61.68%"
        }
    ]
)

st.dataframe(
    attack_comparison,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "DeepFool shows the largest impact in this experiment."
)

st.markdown("### ⚔️ Attack Impact")

attack_visual = pd.DataFrame(
    {
        "Attack": [
            "Clean",
            "FGSM",
            "PGD",
            "DeepFool"
        ],
        "Accuracy": [
            95.04,
            88.78,
            87.83,
            5.37
        ]
    }
)

fig_accuracy = px.bar(
    attack_visual,
    x="Attack",
    y="Accuracy",
    text="Accuracy",
    title="IDS Accuracy Under Adversarial Conditions",
    labels={
        "Accuracy": "Accuracy (%)"
    }
)

fig_accuracy.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig_accuracy.update_layout(
    height=400,
    yaxis_range=[0, 105],
    showlegend=False,
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    )
)

st.plotly_chart(
    fig_accuracy,
    use_container_width=True
)


st.divider()

# ============================================================
# DEFENSE DECISION FRAMEWORK
# ============================================================

st.markdown(
    '<div class="section-title">'
    "Defense Decision Framework"
    "</div>",
    unsafe_allow_html=True
)

st.markdown(
    """
### 🧠 XAI Signals → 🛡️ Defense Decision
"""
)

d1, d2, d3, d4, d5 = st.columns(5)

with d1:
    st.info("**Explanation Drift**")

with d2:
    st.info("**SHAP Similarity**")

with d3:
    st.info("**Feature Changes**")

with d4:
    st.info("**Confidence Change**")

with d5:
    st.info("**Prediction Change**")


st.markdown("### 🛡️ Selected Defense Strategies")

defense1, defense2, defense3 = st.columns(3)

with defense1:
    st.success("**Feature Squeezing**")

with defense2:
    st.success("**Adversarial Training**")

with defense3:
    st.success("**Randomized Smoothing**")

st.caption(
    "These XAI behavior signals will guide the next defense evaluation stage."
)



# ============================================================

# PROJECT PIPELINE

# ============================================================



st.markdown(

    '<div class="section-title">'

    "SentinelXAI Pipeline"

    "</div>",

    unsafe_allow_html=True

)



st.info(

    """

### Project Progress



**✅ Completed**



- ✅ **CICIDS2017 Dataset**

- ✅ **Data Preprocessing**

- ✅ **Deep-Learning IDS**

- ✅ **Adversarial Attack Generation**

- ✅ **SHAP Analysis**

- ✅ **Behavior Feature Extraction**



**🔄 In Progress**



- 🔄 **Defense Decision Framework**



**⬜ Remaining**



- ⬜ **Defense Strategy Evaluation**

- ⬜ **Robustness Evaluation**

"""

)





# ============================================================

# PROJECT COMPLETION DIAGRAM

# ============================================================



st.subheader("📊 Project Completion")



completed_steps = 6

total_steps = 9

completion_percentage = (completed_steps / total_steps) * 100



progress_col1, progress_col2 = st.columns([1, 3])



with progress_col1:



    st.metric(

        "Pipeline Completed",

        f"{completion_percentage:.1f}%"

    )



with progress_col2:



    st.progress(

        completion_percentage / 100

    )



st.caption(

    "6 of 9 major pipeline stages have been completed. "

    "The remaining work focuses on defense evaluation and "

    "robustness testing."

)



# ============================================================
# GENERATED DATA & DEFENSE SIGNALS
# ============================================================

st.markdown(
    '<div class="section-title">'
    "📊 Generated Data & Defense Signals"
    "</div>",
    unsafe_allow_html=True
)

st.caption(
    "Current outputs available for the Defense Decision stage."
)

g1, g2, g3, g4 = st.columns(4)

with g1:

    st.metric(
        "⚡ FGSM",
        "45K",
        "88.78% accuracy"
    )

with g2:

    st.metric(
        "🎯 PGD",
        "45K",
        "87.83% accuracy"
    )

with g3:

    st.metric(
        "💥 DeepFool",
        "45K",
        "5.37% accuracy"
    )

with g4:

    st.metric(
        "🔬 SHAP",
        "500 × 78",
        "3 attack views"
    )


st.markdown("### 🧠 XAI Signals Recorded")

s1, s2, s3, s4 = st.columns(4)

with s1:

    st.metric(
        "Explanation Drift",
        "Recorded"
    )

with s2:

    st.metric(
        "SHAP Similarity",
        "Recorded"
    )

with s3:

    st.metric(
        "Top-10 Overlap",
        "Recorded"
    )

with s4:

    st.metric(
        "Confidence Change",
        "Recorded"
    )


st.markdown("### 🔗 Attack → XAI → Defense")

f1, f2, f3, f4, f5 = st.columns(5)

with f1:

    st.markdown(
        """
        ### ⚔️
        **Attack**

        FGSM  
        PGD  
        DeepFool
        """
    )

with f2:

    st.markdown(
        """
        ### 🧠
        **Model**

        Prediction  
        Confidence
        """
    )

with f3:

    st.markdown(
        """
        ### 🔬
        **SHAP**

        Feature  
        Contributions
        """
    )

with f4:

    st.markdown(
        """
        ### 📈
        **Behavior**

        Drift  
        Similarity  
        Feature Change
        """
    )

with f5:

    st.markdown(
        """
        ### 🛡️
        **Defense**

        Squeezing  
        Adv. Training  
        Smoothing
        """
    )

st.caption(
    "These recorded behavior signals will feed the next defense stage."
)

st.warning(
    "Defense strategy evaluation is still pending: "
    "Feature Squeezing, Adversarial Training and Randomized Smoothing."
)


# FOOTER

# ============================================================





