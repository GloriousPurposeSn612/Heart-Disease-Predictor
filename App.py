from pathlib import Path
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve


# ============================================================
# Final Models Directory
# ============================================================

FINAL_MODELS = Path(__file__).parent / "Final Models"


# ============================================================
# Model Files
# ============================================================

MODEL_FILES = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "KNN": "KNN.pkl",
    "Random Forest Classifier": "Random_Forest_Classifier.pkl"
}


# ============================================================
# Load Final Model Bundles
# ============================================================

models = {}

try:

    for model_name, file_name in MODEL_FILES.items():
        models[model_name] = joblib.load(
            FINAL_MODELS / file_name
        )

except FileNotFoundError as error:

    st.error(
        "A required final model file could not be found. "
        "Please make sure the 'Final Models' folder is present "
        "beside App.py and contains all three .pkl files."
    )

    st.stop()


# ============================================================
# Validate Bundle Structure
# ============================================================

REQUIRED_BUNDLE_KEYS = {
    "model_name",
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "y_test",
    "y_test_pred",
    "y_test_prob",
    "feature_names"
}

for model_name, bundle in models.items():

    missing_keys = REQUIRED_BUNDLE_KEYS.difference(
        bundle.keys()
    )

    if missing_keys:

        st.error(
            f"{model_name} bundle is missing required information: "
            f"{', '.join(sorted(missing_keys))}"
        )

        st.stop()


# ============================================================
# Expected Feature Order
# ============================================================

EXPECTED_FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


# ============================================================
# Validate Feature Order
# ============================================================

for model_name, bundle in models.items():

    if bundle["feature_names"] != EXPECTED_FEATURE_NAMES:

        st.error(
            f"Feature order in the {model_name} bundle does not "
            "match the final notebook feature order."
        )

        st.stop()


# ============================================================
# Training Data Ranges
# ============================================================
# These are the observed ranges in the final notebook dataset.
# The app accepts broader values, but warns when a value falls
# outside the observed training range.

TRAINING_RANGES = {
    "age": (29, 77),
    "trestbps": (94, 200),
    "chol": (126, 564),
    "thalach": (71, 202),
    "oldpeak": (0.0, 6.2)
}


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)


# ============================================================
# Custom Styling
# ============================================================

st.markdown("""
<style>

/* ----------------------------------------------------------
   Remove Streamlit Native Top Header
---------------------------------------------------------- */

[data-testid="stHeader"] {
    display: none;
}


/* ----------------------------------------------------------
   Main Application
---------------------------------------------------------- */

.stApp {
    background: linear-gradient(
        135deg,
        #f5f7fa 0%,
        #e8eef7 100%
    );

    color: #172033;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 7rem;
}


/* ----------------------------------------------------------
   Header
---------------------------------------------------------- */

.main-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    color: #172033 !important;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #4b5563 !important;
    margin-bottom: 28px;
}


/* ----------------------------------------------------------
   Section Titles
---------------------------------------------------------- */

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #172033 !important;
    margin-top: 24px;
    margin-bottom: 10px;
}


/* ----------------------------------------------------------
   General Text
---------------------------------------------------------- */

.stMarkdown,
.stMarkdown p,
.stMarkdown span {
    color: #1f2937;
}


/* ----------------------------------------------------------
   Widget Labels
---------------------------------------------------------- */

[data-testid="stWidgetLabel"] p {
    color: #1f2937 !important;
    font-weight: 600;
}


/* ----------------------------------------------------------
   Selectboxes
---------------------------------------------------------- */

div[data-baseweb="select"] {
    background-color: #ffffff !important;
}

div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #172033 !important;
    border-color: #cbd5e1 !important;
}

div[data-baseweb="select"] span {
    color: #172033 !important;
}


/* Dropdown popup */

div[role="listbox"] {
    background-color: #ffffff !important;
}

div[role="option"] {
    background-color: #ffffff !important;
    color: #172033 !important;
}

div[role="option"]:hover {
    background-color: #eef2f7 !important;
    color: #172033 !important;
}


/* ----------------------------------------------------------
   Sliders
---------------------------------------------------------- */

[data-testid="stSlider"] label {
    color: #1f2937 !important;
}


/* ----------------------------------------------------------
   Buttons
---------------------------------------------------------- */

.stButton > button {
    width: 100%;
    min-height: 52px;

    background-color: #2563eb;
    color: #ffffff !important;

    border: none;
    border-radius: 10px;

    font-size: 17px;
    font-weight: 700;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    color: #ffffff !important;
}


/* ----------------------------------------------------------
   Metrics
---------------------------------------------------------- */

[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #dbe3ef;
    border-radius: 12px;
    padding: 14px;
}

[data-testid="stMetricLabel"] {
    color: #4b5563 !important;
}

[data-testid="stMetricLabel"] p {
    color: #4b5563 !important;
}

[data-testid="stMetricValue"] {
    color: #172033 !important;
}


/* ----------------------------------------------------------
   Project Resources
---------------------------------------------------------- */

.resources-intro {
    color: #4b5563 !important;
    font-size: 16px;
    line-height: 1.6;
    margin-bottom: 14px;
}

.documentation-card {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-left: 5px solid #2563eb;
    border-radius: 12px;
    padding: 18px 20px;
    margin: 8px 0 18px 0;
}

.documentation-title {
    color: #172033 !important;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 7px;
}

.documentation-text {
    color: #4b5563 !important;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 13px;
}

.documentation-link {
    display: inline-block;
    background-color: #2563eb;
    color: #ffffff !important;
    text-decoration: none !important;
    padding: 9px 16px;
    border-radius: 8px;
    font-weight: 700;
}

.documentation-link:hover {
    background-color: #1d4ed8;
    color: #ffffff !important;
}

.resource-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 10px;
}

.resource-card {
    background-color: #ffffff;
    border: 1px solid #dbe3ef;
    border-radius: 10px;
    padding: 14px;
    min-height: 92px;
}

.resource-card-title {
    color: #172033 !important;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 6px;
}

.resource-card-text {
    color: #4b5563 !important;
    font-size: 13px;
    line-height: 1.45;
    margin-bottom: 8px;
}

.resource-card a {
    color: #2563eb !important;
    font-weight: 700;
    text-decoration: none !important;
}

.resource-card a:hover {
    color: #1d4ed8 !important;
    text-decoration: underline !important;
}


/* ----------------------------------------------------------
   Fixed Medical Disclaimer Footer
---------------------------------------------------------- */

.app-footer {
    position: fixed;
    left: 0;
    bottom: 0;

    width: 100%;

    background-color: #fff8dc;

    color: #5b4a00 !important;

    padding: 10px 24px;

    border-top: 1px solid #f1d879;

    box-shadow: 0 -2px 8px rgba(0, 0, 0, 0.08);

    z-index: 999999;

    font-size: 13px;
    line-height: 1.45;
}

.app-footer b {
    color: #4a3b00 !important;
}


/* ----------------------------------------------------------
   Captions
---------------------------------------------------------- */

.stCaption,
[data-testid="stCaptionContainer"] {
    color: #4b5563 !important;
}


/* ----------------------------------------------------------
   DataFrame
---------------------------------------------------------- */

[data-testid="stDataFrame"] {
    background-color: #ffffff;
}


/* ----------------------------------------------------------
   Responsive Resource Layout
---------------------------------------------------------- */

@media (max-width: 900px) {

    .resource-grid {
        grid-template-columns: repeat(2, 1fr);
    }

}

@media (max-width: 600px) {

    .resource-grid {
        grid-template-columns: 1fr;
    }

    .main-title {
        font-size: 36px;
    }

    .subtitle {
        font-size: 16px;
    }

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# Title
# ============================================================

st.markdown(
    '<div class="main-title">❤️ Heart Disease Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Heart Disease Prediction System'
    '</div>',
    unsafe_allow_html=True
)


# =========================
# Global Button Styling
# =========================

st.markdown("""
<style>

div.stLinkButton > a {
    background-color: #000000 !important;
    color: white !important;
    border: none !important;
    transition: background-color 0.5s ease, opacity 0.5s ease;
}

div.stLinkButton > a:hover {
    background-color: #4387D6 !important;
    color: white !important;
    opacity: 0.85 !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# Project Resources & Documentation
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📚 Project Resources & Documentation'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="resources-intro">'
    'Explore the project resources below. If you are unfamiliar '
    'with the medical terminology used in the patient parameters, '
    'please read the documentation first.'
    '</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Documentation Highlight
# ------------------------------------------------------------

st.markdown("""
<div class="documentation-card">

<div class="documentation-title">
📖 Understand The Features First
</div>

<div class="documentation-text">
The parameters below use medical terminology and
dataset-specific categories. Before entering values,
please review the documentation to understand what
each feature represents, its significance in the dataset,
and the distinction between a condition/category and
its encoded value.
</div>

</div>
""", unsafe_allow_html=True)

st.link_button(
    "📖 Read Documentation →",
    "https://github.com/GloriousPurposeSn612/Heart-Disease-Predictor/blob/main/Medical%20Terms.md"
)


# ------------------------------------------------------------
# Additional Resources
# ------------------------------------------------------------

st.markdown("""
<div class="resource-grid">

<div class="resource-card">

<div class="resource-card-title">
💻 GitHub Repository
</div>

<div class="resource-card-text">
Explore the project's source code and implementation.
</div>

<a class="resource-button"
   href="https://github.com/GloriousPurposeSn612/Heart-Disease-Predictor"
   target="_blank">
    View Repository →
</a>
            
</div>


<div class="resource-card">

<div class="resource-card-title">
📊 Dataset
</div>

<div class="resource-card-text">
View the dataset used for model development and evaluation.
</div>

<a class="resource-button"
   href="https://github.com/GloriousPurposeSn612/Heart-Disease-Predictor/blob/main/data.csv"
   target="_blank">
    View Dataset →
</a>

</div>


<div class="resource-card">

<div class="resource-card-title">
🧪 ML Notebook
</div>

<div class="resource-card-text">
Review preprocessing, training, evaluation and visualizations.
</div>

<a class="resource-button"
   href="https://github.com/GloriousPurposeSn612/Heart-Disease-Predictor/blob/main/HDP_Model.ipynb"
   target="_blank">
    Open Notebook →
</a>

</div>


<div class="resource-card">

<div class="resource-card-title">
📋 Technical Documentation
</div>

<div class="resource-card-text">
Learn about the workflow, methodology and implementation.
</div>

<a class="resource-button"
   href="https://github.com/GloriousPurposeSn612/Heart-Disease-Predictor/blob/main/README.md"
   target="_blank">
    Read Documentation →
</a>

</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# Model Selection
# ============================================================

st.markdown(
    '<div class="section-title">Select Prediction Model</div>',
    unsafe_allow_html=True
)

selected_model_name = st.selectbox(
    "Choose a trained model",
    list(models.keys())
)

selected_model_bundle = models[selected_model_name]
selected_model = selected_model_bundle["model"]

st.info(
    "Target class: 0 = No Heart Disease, 1 = Heart Disease."
)


# ============================================================
# Patient Parameters
# ============================================================

st.markdown(
    '<div class="section-title">Patient Parameters</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# Column 1
# ------------------------------------------------------------

with col1:

    age = st.slider(
        "Age",
        min_value=10,
        max_value=100,
        value=55,
        step=1
    )

    sex_label = st.selectbox(
        "Sex",
        [
            "Female",
            "Male"
        ]
    )

    cp_label = st.selectbox(
        "Chest Pain Type",
        [
            "Typical Angina (0)",
            "Atypical Angina (1)",
            "Non-anginal Pain (2)",
            "Asymptomatic (3)"
        ]
    )

    trestbps = st.slider(
        "Resting Blood Pressure (mm Hg)",
        min_value=70,
        max_value=220,
        value=130,
        step=1
    )


# ------------------------------------------------------------
# Column 2
# ------------------------------------------------------------

with col2:

    chol = st.slider(
        "Serum Cholesterol (mg/dl)",
        min_value=100,
        max_value=600,
        value=240,
        step=1
    )

    fbs_label = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [
            "No (0)",
            "Yes (1)"
        ]
    )

    restecg_label = st.selectbox(
        "Resting ECG",
        [
            "Normal (0)",
            "ST-T Wave Abnormality (1)",
            "Left Ventricular Hypertrophy (2)"
        ]
    )

    thalach = st.slider(
        "Maximum Heart Rate",
        min_value=50,
        max_value=220,
        value=150,
        step=1
    )


# ------------------------------------------------------------
# Column 3
# ------------------------------------------------------------

with col3:

    exang_label = st.selectbox(
        "Exercise Induced Angina",
        [
            "No (0)",
            "Yes (1)"
        ]
    )

    oldpeak = st.slider(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=7.0,
        value=1.0,
        step=0.1
    )

    slope_label = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        [
            "Upsloping (0)",
            "Flat (1)",
            "Downsloping (2)"
        ]
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        [0, 1, 2, 3, 4]
    )

    thal_label = st.selectbox(
        "Thal Encoded Category",
        [
            "Code 0",
            "Normal (1)",
            "Fixed Defect (2)",
            "Reversible Defect (3)"
        ]
    )


# ============================================================
# Convert UI Labels Into Dataset Encoding
# ============================================================

sex = 1 if sex_label == "Male" else 0

cp_map = {
    "Typical Angina (0)": 0,
    "Atypical Angina (1)": 1,
    "Non-anginal Pain (2)": 2,
    "Asymptomatic (3)": 3
}

cp = cp_map[cp_label]

fbs = 1 if fbs_label.startswith("Yes") else 0

restecg_map = {
    "Normal (0)": 0,
    "ST-T Wave Abnormality (1)": 1,
    "Left Ventricular Hypertrophy (2)": 2
}

restecg = restecg_map[restecg_label]

exang = 1 if exang_label.startswith("Yes") else 0

slope_map = {
    "Upsloping (0)": 0,
    "Flat (1)": 1,
    "Downsloping (2)": 2
}

slope = slope_map[slope_label]

thal_map = {
    "Code 0": 0,
    "Normal (1)": 1,
    "Fixed Defect (2)": 2,
    "Reversible Defect (3)": 3
}

thal = thal_map[thal_label]


# ============================================================
# Check For Inputs Outside Training Data Range
# ============================================================

out_of_training_range = []

if not (
    TRAINING_RANGES["age"][0]
    <= age
    <= TRAINING_RANGES["age"][1]
):
    out_of_training_range.append(
        f"Age: training range was "
        f"{TRAINING_RANGES['age'][0]}-{TRAINING_RANGES['age'][1]}"
    )

if not (
    TRAINING_RANGES["trestbps"][0]
    <= trestbps
    <= TRAINING_RANGES["trestbps"][1]
):
    out_of_training_range.append(
        f"Resting BP: training range was "
        f"{TRAINING_RANGES['trestbps'][0]}-"
        f"{TRAINING_RANGES['trestbps'][1]} mm Hg"
    )

if not (
    TRAINING_RANGES["chol"][0]
    <= chol
    <= TRAINING_RANGES["chol"][1]
):
    out_of_training_range.append(
        f"Cholesterol: training range was "
        f"{TRAINING_RANGES['chol'][0]}-"
        f"{TRAINING_RANGES['chol'][1]} mg/dl"
    )

if not (
    TRAINING_RANGES["thalach"][0]
    <= thalach
    <= TRAINING_RANGES["thalach"][1]
):
    out_of_training_range.append(
        f"Maximum heart rate: training range was "
        f"{TRAINING_RANGES['thalach'][0]}-"
        f"{TRAINING_RANGES['thalach'][1]}"
    )

if not (
    TRAINING_RANGES["oldpeak"][0]
    <= oldpeak
    <= TRAINING_RANGES["oldpeak"][1]
):
    out_of_training_range.append(
        f"Oldpeak: training range was "
        f"{TRAINING_RANGES['oldpeak'][0]}-"
        f"{TRAINING_RANGES['oldpeak'][1]}"
    )


if out_of_training_range:

    st.markdown("""
                <div class="disclaimer">
                <b>Note:</b>
                One or more values are outside the ranges observed 
                in the training dataset. The model can still generate 
                a prediction, but the result involves extrapolation 
                beyond the training-data range.
                
                </div>
                """, unsafe_allow_html=True)

    for message in out_of_training_range:
        st.caption(f"- {message}")


# ============================================================
# Prepare Model Input
# ============================================================

input_data = pd.DataFrame(
    [[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]],
    columns=selected_model_bundle["feature_names"]
)


# ============================================================
# Prediction
# ============================================================

st.write("")

if st.button(
    "Predict Heart Disease",
    width='stretch'
):

    prediction = selected_model.predict(
        input_data
    )[0]

    st.markdown(
        '<div class="section-title">Prediction Result</div>',
        unsafe_allow_html=True
    )


    if prediction == 1:

        st.error(
            "Prediction: Heart Disease (Class 1)"
        )

    else:

        st.success(
            "Prediction: No Heart Disease (Class 0)"
        )


    # --------------------------------------------------------
    # Model-Estimated Probability
    # --------------------------------------------------------

    if hasattr(
        selected_model,
        "predict_proba"
    ):

        probability = selected_model.predict_proba(
            input_data
        )[0][1]

        probability_col1, probability_col2 = st.columns(2)

        with probability_col1:

            st.metric(
                "Model-Estimated Probability of Class 1",
                f"{probability:.2%}"
            )

        with probability_col2:

            st.write(
                "Probability Visualization"
            )

            st.progress(
                min(
                    max(probability, 0.0),
                    1.0
                )
            )

    else:

        st.info(
            "This model does not provide a direct probability "
            "estimate."
        )


    # --------------------------------------------------------
    # Input Summary
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Input Summary</div>',
        unsafe_allow_html=True
    )

    display_input = pd.DataFrame({

        "Parameter": [
            "Age",
            "Sex",
            "Chest Pain Type",
            "Resting Blood Pressure",
            "Cholesterol",
            "Fasting Blood Sugar > 120",
            "Resting ECG",
            "Maximum Heart Rate",
            "Exercise Induced Angina",
            "ST Depression",
            "Slope",
            "Major Vessels",
            "Thal Encoded Category"
        ],

        "Value": [
            str(age),
            sex_label,
            cp_label,
            str(trestbps),
            str(chol),
            fbs_label,
            restecg_label,
            str(thalach),
            exang_label,
            str(oldpeak),
            slope_label,
            str(ca),
            thal_label
        ]
    })

    st.dataframe(
        display_input,
        width='stretch',
        hide_index=True
    )


# ============================================================
# Model Performance
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    'Selected Model Performance'
    '</div>',
    unsafe_allow_html=True
)

metrics = {
    "Accuracy": selected_model_bundle["accuracy"],
    "Precision": selected_model_bundle["precision"],
    "Recall": selected_model_bundle["recall"],
    "F1 Score": selected_model_bundle["f1_score"],
    "ROC-AUC": selected_model_bundle["roc_auc"]
}

metric_columns = st.columns(5)

for column, (
    metric_name,
    metric_value
) in zip(
    metric_columns,
    metrics.items()
):

    with column:

        st.metric(
            metric_name,
            f"{metric_value:.2%}"
        )


# ============================================================
# Selected Model Performance Visualization
# ============================================================

fig_performance, ax_performance = plt.subplots(
    figsize=(10, 4.5)
)

performance_names = list(metrics.keys())
performance_values = list(metrics.values())

bars = ax_performance.bar(
    performance_names,
    performance_values
)

ax_performance.set_title(
    f"{selected_model_name} - Test Set Performance"
)

ax_performance.set_ylabel(
    "Score"
)

ax_performance.set_ylim(
    0,
    1.05
)

ax_performance.tick_params(
    axis="x",
    rotation=20
)

for bar, value in zip(
    bars,
    performance_values
):

    ax_performance.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

fig_performance.tight_layout()

st.pyplot(
    fig_performance,
    clear_figure=True
)


# ============================================================
# Confusion Matrix + ROC Curve
# ============================================================

visual_col1, visual_col2 = st.columns(2)


# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------

with visual_col1:

    st.markdown(
        '<div class="section-title">'
        'Confusion Matrix'
        '</div>',
        unsafe_allow_html=True
    )

    y_test = selected_model_bundle["y_test"]
    y_test_pred = selected_model_bundle["y_test_pred"]

    confusion = confusion_matrix(
        y_test,
        y_test_pred
    )

    confusion_annotations = [
        [
            f"{confusion[0, 0]}\nTN",
            f"{confusion[0, 1]}\nFP"
        ],
        [
            f"{confusion[1, 0]}\nFN",
            f"{confusion[1, 1]}\nTP"
        ]
    ]

    fig_confusion, ax_confusion = plt.subplots(
        figsize=(6, 5)
    )

    sns.heatmap(
        confusion,
        annot=confusion_annotations,
        fmt="",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=1,
        linecolor="white",
        xticklabels=[
            "No Disease (0)",
            "Disease (1)"
        ],
        yticklabels=[
            "No Disease (0)",
            "Disease (1)"
        ],
        ax=ax_confusion
    )

    ax_confusion.set_xlabel(
        "Predicted Label"
    )

    ax_confusion.set_ylabel(
        "Actual Label"
    )

    ax_confusion.set_title(
        "Test Set Confusion Matrix"
    )

    fig_confusion.tight_layout()

    st.pyplot(
        fig_confusion,
        clear_figure=True
    )


# ------------------------------------------------------------
# ROC Curve
# ------------------------------------------------------------

with visual_col2:

    st.markdown(
        '<div class="section-title">'
        'ROC Curve'
        '</div>',
        unsafe_allow_html=True
    )

    y_test_prob = selected_model_bundle["y_test_prob"]

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        y_test_prob
    )

    fig_roc, ax_roc = plt.subplots(
        figsize=(6, 5)
    )

    ax_roc.plot(
        false_positive_rate,
        true_positive_rate,
        label=(
            f"ROC-AUC = "
            f"{selected_model_bundle['roc_auc']:.3f}"
        )
    )

    ax_roc.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax_roc.set_xlabel(
        "False Positive Rate"
    )

    ax_roc.set_ylabel(
        "True Positive Rate"
    )

    ax_roc.set_title(
        "Test Set ROC Curve"
    )

    ax_roc.set_xlim(
        0,
        1
    )

    ax_roc.set_ylim(
        0,
        1
    )

    ax_roc.legend(
        loc="lower right"
    )

    fig_roc.tight_layout()

    st.pyplot(
        fig_roc,
        clear_figure=True
    )


# ============================================================
# Model-Specific Interpretation
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    'Model-Specific Interpretation'
    '</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

if (
    selected_model_name
    == "Random Forest Classifier"
    and hasattr(
        selected_model,
        "feature_importances_"
    )
):

    feature_importance = pd.DataFrame({
        "Feature": selected_model_bundle["feature_names"],
        "Importance": selected_model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=True
    )

    fig_importance, ax_importance = plt.subplots(
        figsize=(9, 5)
    )

    ax_importance.barh(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )

    ax_importance.set_xlabel(
        "Importance"
    )

    ax_importance.set_ylabel(
        "Feature"
    )

    ax_importance.set_title(
        "Random Forest Model-Derived Feature Importance"
    )

    fig_importance.tight_layout()

    st.pyplot(
        fig_importance,
        clear_figure=True
    )

    st.info(
        "Feature importance describes how the trained Random "
        "Forest model uses the features. It does not establish "
        "medical causation or clinical importance."
    )


# ------------------------------------------------------------
# Logistic Regression
# ------------------------------------------------------------

elif selected_model_name == "Logistic Regression":

    st.info(
        "Logistic Regression does not use a tree-based feature "
        "importance measure. Its coefficients are scale-dependent "
        "in this project because the final Logistic Regression "
        "model was trained on the original feature scales. "
        "Therefore, a feature-importance chart is not shown."
    )


# ------------------------------------------------------------
# KNN
# ------------------------------------------------------------

elif selected_model_name == "KNN":

    st.info(
        "K-Nearest Neighbors is a distance-based classifier and "
        "does not provide a native feature-importance vector. "
        "Therefore, a feature-importance chart is not shown."
    )


# ============================================================
# Fixed Medical Disclaimer Footer
# ============================================================

st.markdown("""
<div class="app-footer">

<b>Medical Disclaimer:</b>
This application is an academic machine learning project
intended for educational and demonstration purposes only.
It is not a medical diagnostic tool and should not be used
as a substitute for professional medical advice, diagnosis,
or treatment.

</div>
""", unsafe_allow_html=True)