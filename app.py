"""CardioScope AI: Heart Disease Risk Prediction Interactive Analytics Dashboard.

A modern, responsive, clinical-inspired Streamlit dashboard providing
demographic and diagnostic risk assessments, model benchmark comparisons,
exploratory data visualizations, and educational decision-support insights.
"""

import json
import os
import sys
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Configure page metadata and layout
st.set_page_config(
    page_title="CardioScope AI — Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Resolve project root and paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    DEFAULT_DATA_PATH,
    EXPECTED_FEATURE_COLUMNS,
    load_heart_disease_data,
    get_dataset_summary,
)
from src.predict import HeartDiseasePredictor, CLINICAL_DISCLAIMER
from src.preprocessing import FEATURE_RANGES

# Custom CSS for a refined, professional healthcare analytics interface
CUSTOM_CSS = """
<style>
    /* Global font and background styling */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Clean container backgrounds */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 14px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
        border-color: #cbd5e1;
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #0284c7;
        margin-top: 4px;
        font-weight: 500;
    }
    
    /* Medical Alert Callouts */
    .disclaimer-banner {
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #166534;
        margin-bottom: 20px;
    }
    .clinical-caution {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #92400e;
        margin-top: 15px;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 10px;
        margin-bottom: 16px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
    }
    
    /* Result Box Styling */
    .result-box-low {
        background-color: #ecfdf5;
        border: 2px solid #10b981;
        border-radius: 10px;
        padding: 22px;
        margin-bottom: 20px;
    }
    .result-box-elevated {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 22px;
        margin-bottom: 20px;
    }
    .result-box-borderline {
        background-color: #fffbeb;
        border: 2px solid #f59e0b;
        border-radius: 10px;
        padding: 22px;
        margin-bottom: 20px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_predictor_instance():
    """Cache and load the prediction inference engine."""
    try:
        return HeartDiseasePredictor()
    except Exception as e:
        return None


@st.cache_data
def get_cached_dataset():
    """Cache and load dataset and summary stats."""
    df = load_heart_disease_data()
    summary = get_dataset_summary(df)
    return df, summary


@st.cache_data
def get_evaluation_results():
    """Load model comparison CSV results."""
    csv_path = os.path.join(PROJECT_ROOT, "reports", "model_comparison_results.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path).set_index("model")
    return None


predictor = get_predictor_instance()
df_data, data_summary = get_cached_dataset()
comp_results = get_evaluation_results()

# ==============================================================================
# SIDEBAR NAVIGATION & CLINICAL CONTEXT
# ==============================================================================
with st.sidebar:
    st.markdown("## ❤️ **CardioScope AI**")
    st.markdown("*Clinical Machine Learning Decision Support Prototype*")
    st.divider()

    selected_page = st.radio(
        "Navigation Menu",
        [
            "📊 Executive Dashboard",
            "🩺 Patient Risk Assessment",
            "📈 Model Benchmark & Evaluation",
            "🔬 Dataset Insights & EDA",
            "ℹ️ Methodology & Disclaimers",
        ],
        index=0,
    )

    st.divider()
    st.markdown("### 🏥 System Status")
    if predictor and predictor.pipeline:
        champion_name = predictor.metadata.get("best_model", {}).get("name", "Logistic Regression")
        st.success(f"**Engine Active:** {champion_name}")
        st.caption(f"Cohort Size: {data_summary['total_records']} instances")
    else:
        st.error("⚠️ Model artifacts missing! Run `python src/train.py`.")

    st.markdown("---")
    st.caption(
        "**Clinical Governance Disclaimer:**\n"
        "This system is strictly an academic decision-support prototype. "
        "Outputs indicate statistical dataset correlation and never replace formal clinical diagnosis."
    )


# ==============================================================================
# PAGE 1: EXECUTIVE DASHBOARD OVERVIEW
# ==============================================================================
if selected_page == "📊 Executive Dashboard":
    st.title("🫀 Heart Disease Risk Prediction Dashboard")
    st.markdown(
        "Cardiovascular diseases represent the single largest contributor to global morbidity. "
        "This analytics platform leverages machine-learning classification models trained on the "
        "benchmark **UCI Cleveland Heart Disease Cohort** to detect patterns of angiographically "
        "confirmed coronary artery disease ($> 50\\%$ lumen stenosis)."
    )

    st.markdown(
        """
        <div class="disclaimer-banner">
            <strong>Clinical Prototype Notice:</strong> Designed for medical engineering education, 
            decision-support research, and portfolio demonstration. Not FDA/CE-cleared for autonomous diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key Performance Indicator Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Validation Cohort</div>
                <div class="metric-value">{data_summary['total_records']}</div>
                <div class="metric-sub">{data_summary['disease_percentage']}% Angiographic Positive</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        champ_sens = "92.86%"
        if comp_results is not None and "recall" in comp_results.columns:
            champ_sens = f"{comp_results.loc['Logistic Regression', 'recall']:.1%}"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Champion Sensitivity</div>
                <div class="metric-value">{champ_sens}</div>
                <div class="metric-sub">Only 2 Missed Cases (FN)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        champ_auc = "0.9621"
        if comp_results is not None and "roc_auc" in comp_results.columns:
            champ_auc = f"{comp_results.loc['Logistic Regression', 'roc_auc']:.4f}"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Discrimination (ROC-AUC)</div>
                <div class="metric-value">{champ_auc}</div>
                <div class="metric-sub">Held-Out Test Set</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Benchmarked Models</div>
                <div class="metric-value">4</div>
                <div class="metric-sub">5-Fold Stratified CV</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">Champion Model Architecture</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown(
            """
            ### Regularized Logistic Regression ($L_2$ Penalty)
            The production pipeline employs an optimized **Scikit-Learn ColumnTransformer + Regularized Logistic Regression** model.
            
            **Why Logistic Regression was selected over black-box ensembles:**
            - **Minimizes False Negatives:** Achieved **92.86% Sensitivity** (Recall), missing only 2 diseased cases out of 28 in held-out testing.
            - **Superior Generalization:** Scored the highest test ROC-AUC (**0.9621**), demonstrating exceptional discrimination without overfitting.
            - **Clinical Interpretability:** Transparent model coefficients directly reflect established cardiovascular physiology (e.g., fluoroscopy vessels, ST depression, exercise-induced angina).
            - **Calibration:** Produces continuous probabilities that align monotonically with clinical disease severity.
            """
        )
    with c2:
        if os.path.exists(os.path.join(PROJECT_ROOT, "reports", "figures", "model_comparison.png")):
            st.image(
                os.path.join(PROJECT_ROOT, "reports", "figures", "model_comparison.png"),
                caption="Held-out Test Performance Across Evaluated ML Models",
                use_container_width=True,
            )

    st.markdown('<div class="section-header">End-to-End Decision Support Workflow</div>', unsafe_allow_html=True)
    step_col1, step_col2, step_col3, step_col4 = st.columns(4)
    with step_col1:
        st.info("**1. Data Input & Validation**\n\nPatient clinical vitals, symptom history, and stress ECG indicators entered via structured form.")
    with step_col2:
        st.info("**2. Preprocessing Pipeline**\n\nMedian/mode imputation, standard scaling, and one-hot encoding fit strictly on training splits.")
    with step_col3:
        st.info("**3. ML Classification**\n\nTuned classifier computes calibrated risk score and angiographic disease classification.")
    with step_col4:
        st.info("**4. Interpretability Report**\n\nIndividualized risk triggers highlighted with non-invasive follow-up recommendations.")


# ==============================================================================
# PAGE 2: PATIENT RISK ASSESSMENT (PREDICTOR)
# ==============================================================================
elif selected_page == "🩺 Patient Risk Assessment":
    st.title("🩺 Patient Health Indicator Risk Assessment")
    st.markdown(
        "Input patient demographic information, physiological vital signs, stress-test responses, "
        "and diagnostic imaging findings to evaluate statistical risk of angiographic coronary disease."
    )

    if predictor is None:
        st.error("Prediction engine not initialized. Please run `python src/train.py`.")
        st.stop()

    # Preset profiles to expedite demonstration
    preset_choice = st.selectbox(
        "Load Clinical Profile Preset (or customize below):",
        [
            "Custom Patient Input (Manual)",
            "Preset A: Healthy Baseline Patient (Low Risk Profile)",
            "Preset B: Borderline Adult with Moderate Risk Factors",
            "Preset C: Symptomatic Patient with Multi-Vessel Disease (High Risk Profile)",
        ],
    )

    # Profile defaults
    default_vals = {
        "age": 52, "sex": 1, "cp": 2, "trestbps": 125, "chol": 210, "fbs": 0,
        "restecg": 0, "thalach": 160, "exang": 0, "oldpeak": 0.4, "slope": 1,
        "ca": 0, "thal": 3,
    }

    if preset_choice == "Preset A: Healthy Baseline Patient (Low Risk Profile)":
        default_vals = {
            "age": 36, "sex": 0, "cp": 2, "trestbps": 115, "chol": 178, "fbs": 0,
            "restecg": 0, "thalach": 175, "exang": 0, "oldpeak": 0.0, "slope": 1,
            "ca": 0, "thal": 3,
        }
    elif preset_choice == "Preset B: Borderline Adult with Moderate Risk Factors":
        default_vals = {
            "age": 58, "sex": 1, "cp": 3, "trestbps": 140, "chol": 242, "fbs": 0,
            "restecg": 1, "thalach": 140, "exang": 0, "oldpeak": 1.2, "slope": 2,
            "ca": 1, "thal": 3,
        }
    elif preset_choice == "Preset C: Symptomatic Patient with Multi-Vessel Disease (High Risk Profile)":
        default_vals = {
            "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 1,
            "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 2.6, "slope": 2,
            "ca": 3, "thal": 7,
        }

    with st.form("patient_assessment_form"):
        st.markdown("#### Patient Health Profile Parameters")
        form_col1, form_col2, form_col3 = st.columns(3)

        with form_col1:
            st.markdown("##### 👤 Demographics & Vitals")
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=default_vals["age"], help="Patient age in completed years.")
            sex_label = st.radio("Biological Sex", ["Female", "Male"], index=default_vals["sex"], horizontal=True)
            sex = 1 if sex_label == "Male" else 0
            trestbps = st.number_input("Resting Systolic BP (mm Hg)", min_value=80, max_value=240, value=default_vals["trestbps"], help="Hospital admission resting systolic blood pressure. Normal is < 120 mm Hg.")
            chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=default_vals["chol"], help="Total fasting serum cholesterol. Desirable is < 200 mg/dL.")
            fbs_label = st.radio("Fasting Blood Sugar > 120 mg/dL", ["No (<= 120 mg/dL)", "Yes (> 120 mg/dL)"], index=default_vals["fbs"], horizontal=True)
            fbs = 1 if "Yes" in fbs_label else 0

        with form_col2:
            st.markdown("##### 🏃 Cardiac Stress Symptoms")
            cp_options = {
                1: "1: Typical Angina",
                2: "2: Atypical Angina",
                3: "3: Non-Anginal Pain",
                4: "4: Asymptomatic (Silent)",
            }
            cp_idx = [1, 2, 3, 4].index(default_vals["cp"])
            cp_selected = st.selectbox("Chest Pain Presentation", list(cp_options.values()), index=cp_idx, help="Type of chest pain reported. Note: Asymptomatic often reflects silent multi-vessel disease in this cohort.")
            cp = [k for k, v in cp_options.items() if v == cp_selected][0]

            thalach = st.number_input("Max Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=default_vals["thalach"], help="Peak heart rate achieved during treadmill exercise stress testing.")
            exang_label = st.radio("Exercise-Induced Angina", ["No", "Yes"], index=default_vals["exang"], horizontal=True, help="Angina symptoms provoked during exertion.")
            exang = 1 if exang_label == "Yes" else 0

            slope_options = {
                1: "1: Upsloping (Favorable)",
                2: "2: Flat (Equivocal/Ischemic)",
                3: "3: Downsloping (Severe Ischemia)",
            }
            slope_idx = [1, 2, 3].index(default_vals["slope"])
            slope_selected = st.selectbox("Peak Exercise ST Slope", list(slope_options.values()), index=slope_idx)
            slope = [k for k, v in slope_options.items() if v == slope_selected][0]

        with form_col3:
            st.markdown("##### 🔬 ECG & Advanced Imaging")
            restecg_options = {
                0: "0: Normal",
                1: "1: ST-T Wave Abnormality",
                2: "2: Left Ventricular Hypertrophy (LVH)",
            }
            restecg_idx = [0, 1, 2].index(default_vals["restecg"])
            restecg_selected = st.selectbox("Resting Electrocardiogram", list(restecg_options.values()), index=restecg_idx)
            restecg = [k for k, v in restecg_options.items() if v == restecg_selected][0]

            oldpeak = st.number_input("ST Depression / Oldpeak (mm)", min_value=0.0, max_value=10.0, value=float(default_vals["oldpeak"]), step=0.1, help="ST-segment depression induced by exercise relative to rest.")
            ca = st.slider("Major Vessels Colored by Fluoroscopy (0 - 3)", min_value=0, max_value=3, value=int(default_vals["ca"]), help="Number of major coronary arteries opacified by catheter fluoroscopy.")

            thal_options = {
                3: "3: Normal Perfusion",
                6: "6: Fixed Defect (Prior Infarct)",
                7: "7: Reversible Defect (Inducible Ischemia)",
            }
            thal_keys = [3, 6, 7]
            thal_idx = thal_keys.index(int(default_vals["thal"]))
            thal_selected = st.selectbox("Thallium-201 Stress Scintigraphy", list(thal_options.values()), index=thal_idx)
            thal = [k for k, v in thal_options.items() if v == thal_selected][0]

        submit_btn = st.form_submit_button("⚡ Evaluate Heart Disease Risk", use_container_width=True, type="primary")

    if submit_btn:
        patient_dict = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": float(ca),
            "thal": float(thal),
        }

        with st.spinner("Analyzing physiological patterns with ML pipeline..."):
            try:
                assessment = predictor.predict_patient(patient_dict)
            except Exception as exc:
                st.error(f"Inference error: {exc}")
                st.stop()

        st.markdown('<div class="section-header">Assessment Results & Clinical Interpretation</div>', unsafe_allow_html=True)

        res_box_class = "result-box-low"
        if assessment["risk_category"] == "Elevated Clinical Likelihood":
            res_box_class = "result-box-elevated"
        elif assessment["risk_category"] == "Moderate / Borderline Clinical Likelihood":
            res_box_class = "result-box-borderline"

        st.markdown(
            f"""
            <div class="{res_box_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h3 style="margin: 0; color: {assessment['risk_color']};">
                            {assessment['risk_category']}
                        </h3>
                        <p style="margin: 5px 0 0 0; font-size: 1.05rem; font-weight: 600; color: #1e293b;">
                            {assessment['prediction_label']}
                        </p>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 2.2rem; font-weight: 800; color: {assessment['risk_color']};">
                            {assessment['risk_percentage']}%
                        </span>
                        <div style="font-size: 0.8rem; color: #64748b; font-weight: 500;">Calculated Model Risk Score</div>
                    </div>
                </div>
                <hr style="border: 0; border-top: 1px solid rgba(0,0,0,0.1); margin: 15px 0;">
                <p style="margin: 0; font-size: 0.95rem; color: #334155;">
                    <strong>Model Finding:</strong> {assessment['interpretation']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        r_col1, r_col2 = st.columns([1.2, 1])
        with r_col1:
            st.markdown("#### 🔍 Individualized Contributing Factors")
            factors = assessment["contributing_factors"]
            if factors:
                for f in factors:
                    severity_badge = "🚨" if f["severity"] == "High" else "⚠️" if f["severity"] == "Moderate" else "ℹ️"
                    with st.expander(f"{severity_badge} **{f['factor']}** — *{f['finding']}*", expanded=True):
                        st.write(f["detail"])
            else:
                st.success("✅ No elevated acute ischemic or fluoroscopic risk factors identified. Vital signs and stress responses fall within non-diseased benchmark parameters.")

        with r_col2:
            st.markdown("#### 📋 Recommended Decision-Support Next Steps")
            st.markdown(
                """
                - **Primary Consultation:** Review findings with an attending cardiologist or primary physician.
                - **Confirmatory Diagnostics:** Consider follow-up non-invasive testing (Stress Echocardiogram or Coronary CT Angiogram) if symptoms persist.
                - **Cardiovascular Lifestyle:** Support arterial health via Mediterranean dietary patterns, routine aerobic activity, and blood pressure monitoring.
                - **Follow-up Protocol:** Re-evaluate lipid profile and fasting glucose annually.
                """
            )

        st.markdown(
            f"""
            <div class="clinical-caution">
                <strong>Medical Disclaimer:</strong> {assessment['disclaimer']}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# PAGE 3: MODEL BENCHMARK & EVALUATION
# ==============================================================================
elif selected_page == "📈 Model Benchmark & Evaluation":
    st.title("📈 Machine Learning Model Benchmark & Clinical Evaluation")
    st.markdown(
        "Compare classification models evaluated on the **held-out test set** ($n = 61$, 28 positive, 33 negative) "
        "following 5-fold stratified cross-validation on the training set ($n = 242$)."
    )

    if comp_results is not None:
        st.markdown("### 🏆 Comprehensive Model Comparison Table")
        st.dataframe(
            comp_results.style.format({
                "accuracy": "{:.2%}",
                "recall": "{:.2%}",
                "specificity": "{:.2%}",
                "precision": "{:.2%}",
                "f1_score": "{:.4f}",
                "roc_auc": "{:.4f}",
                "pr_auc": "{:.4f}",
                "cv_roc_auc": "{:.4f}",
                "cv_recall": "{:.4f}",
                "false_negatives": "{:d}",
                "false_positives": "{:d}",
            }).highlight_max(subset=["accuracy", "recall", "specificity", "f1_score", "roc_auc"], color="#dcfce7"),
            use_container_width=True,
        )

        st.markdown(
            """
            *Note: In medical screening, **Recall / Sensitivity** is prioritized over raw accuracy because a **False Negative** (missing a diseased patient) is far more hazardous than a False Positive.*
            """
        )

    st.markdown('<div class="section-header">Diagnostic Evaluation Charts</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        cm_path = os.path.join(PROJECT_ROOT, "reports", "figures", "confusion_matrices.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix Grid Across Evaluated Models", use_container_width=True)

    with c2:
        roc_path = os.path.join(PROJECT_ROOT, "reports", "figures", "roc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, caption="Receiver Operating Characteristic (ROC) Discrimination Curves", use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        pr_path = os.path.join(PROJECT_ROOT, "reports", "figures", "pr_curves.png")
        if os.path.exists(pr_path):
            st.image(pr_path, caption="Precision-Recall Curves relative to Sample Prevalence", use_container_width=True)

    with c4:
        fi_path = os.path.join(PROJECT_ROOT, "reports", "figures", "feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, caption="Top Standardized Predictive Coefficients / Feature Importances", use_container_width=True)

    st.markdown('<div class="section-header">Medical Rationale: False Negatives vs False Positives</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **The False Negative Dilemma ($FN$):** When an algorithm falsely labels a patient with coronary disease as healthy, the patient is reassured erroneously and denied cardioprotective medications or invasive intervention, risking unheralded myocardial infarction or sudden cardiac death.
        - **The False Positive Dilemma ($FP$):** When an algorithm flags a healthy person, they receive non-invasive confirmatory testing (e.g. stress echo or CT angiography). While this introduces transient anxiety and testing costs, it does not carry mortal clinical consequences.
        - **Champion Selection Conclusion:** **Logistic Regression** achieved the highest test Sensitivity (**92.86%**, with only 2 False Negatives) and test ROC-AUC (**0.9621**), making it the clinically superior model.
        """
    )


# ==============================================================================
# PAGE 4: DATASET INSIGHTS & EDA
# ==============================================================================
elif selected_page == "🔬 Dataset Insights & EDA":
    st.title("🔬 Exploratory Data Analysis & Physiological Insights")
    st.markdown(
        "Interactive exploration of patient demographics, vital indicators, and angiographic findings "
        "within the **303-patient Cleveland Heart Disease Cohort**."
    )

    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["📊 Demographic & Hemodynamic Distributions", "🫀 Stress ECG & Imaging", "🔗 Correlation Heatmap"])

    with eda_tab1:
        st.markdown("#### Patient Age, Blood Pressure, and Serum Cholesterol")
        col1, col2 = st.columns(2)

        with col1:
            fig_age, ax_age = plt.subplots(figsize=(6, 4), dpi=150)
            sns.histplot(data=df_data, x="age", hue="target", kde=True, palette={0: "#0284c7", 1: "#e11d48"}, ax=ax_age, bins=15, alpha=0.6)
            ax_age.set_title("Age Distribution by Heart Disease Status", fontsize=11, fontweight="bold")
            ax_age.set_xlabel("Age (years)")
            ax_age.legend(["Presence (1)", "Absence (0)"])
            st.pyplot(fig_age)
            plt.close(fig_age)

        with col2:
            fig_chol, ax_chol = plt.subplots(figsize=(6, 4), dpi=150)
            sns.boxplot(data=df_data, x="target", y="chol", palette=["#0284c7", "#e11d48"], ax=ax_chol)
            ax_chol.set_title("Serum Cholesterol (mg/dL) by Status", fontsize=11, fontweight="bold")
            ax_chol.set_xticklabels(["Absence (0)", "Presence (1)"])
            ax_chol.set_ylabel("Cholesterol (mg/dL)")
            st.pyplot(fig_chol)
            plt.close(fig_chol)

    with eda_tab2:
        st.markdown("#### Stress Test Responses and Fluoroscopy")
        col3, col4 = st.columns(2)

        with col3:
            fig_oldpeak, ax_oldpeak = plt.subplots(figsize=(6, 4), dpi=150)
            sns.boxplot(data=df_data, x="target", y="oldpeak", palette=["#0284c7", "#e11d48"], ax=ax_oldpeak)
            ax_oldpeak.set_title("Exercise ST Depression (Oldpeak mm) by Status", fontsize=11, fontweight="bold")
            ax_oldpeak.set_xticklabels(["Absence (0)", "Presence (1)"])
            ax_oldpeak.set_ylabel("ST Depression (mm)")
            st.pyplot(fig_oldpeak)
            plt.close(fig_oldpeak)

        with col4:
            fig_ca, ax_ca = plt.subplots(figsize=(6, 4), dpi=150)
            ca_ct = pd.crosstab(df_data["ca"], df_data["target"], normalize="index") * 100
            ca_ct.plot(kind="bar", color=["#0284c7", "#e11d48"], ax=ax_ca, edgecolor="#ffffff")
            ax_ca.set_title("Disease Prevalence by Fluoroscopy Vessels (ca)", fontsize=11, fontweight="bold")
            ax_ca.set_xlabel("Vessels Colored (0 - 3)")
            ax_ca.set_ylabel("Prevalence (%)")
            ax_ca.legend(["Absence (0)", "Presence (1)"])
            st.pyplot(fig_ca)
            plt.close(fig_ca)

    with eda_tab3:
        st.markdown("#### Clinical Feature Correlation Matrix")
        numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "target"]
        corr = df_data[numeric_cols].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6), dpi=150)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, square=True, ax=ax_corr, linewidths=0.5)
        ax_corr.set_title("Pairwise Pearson Correlations with Heart Disease Target", fontsize=12, fontweight="bold", pad=10)
        st.pyplot(fig_corr)
        plt.close(fig_corr)


# ==============================================================================
# PAGE 5: METHODOLOGY & CLINICAL DISCLAIMERS
# ==============================================================================
elif selected_page == "ℹ️ Methodology & Disclaimers":
    st.title("ℹ️ Project Methodology & Technical Documentation")

    st.markdown(
        """
        ### Project Background & Purpose
        This project was engineered to deliver a complete, reproducible, healthcare-grade machine-learning
        pipeline for predicting coronary artery disease risk. Inspired by clinical risk prediction architectures,
        it demonstrates end-to-end best practices in data validation, data-leakage prevention, multi-model
        benchmarking, and user-centric decision-support design.
        
        ---
        
        ### Architecture & Implementation Highlights
        - **Pipeline Encapsulation:** Data imputation, one-hot encoding, and standard scaling are assembled into Scikit-Learn `Pipeline` and `ColumnTransformer` objects, fit strictly on training splits to prevent data leakage.
        - **Stratified Splitting:** 80/20 train/test split maintains identical target class proportions in both subsets.
        - **Hyperparameter Tuning:** 5-fold Stratified Cross-Validation on the training partition optimizes hyperparameters without touching the held-out test set.
        - **Defensive Input Validation:** Incoming inference payloads are verified against strict physiological ranges before evaluation.
        - **Comprehensive Automated Testing:** 19 automated unit and integration tests verifying schema validation, preprocessing integrity, inference consistency, and failure modes.
        
        ---
        
        ### Technology Stack
        - **Language & Runtime:** Python 3.13
        - **Data Processing:** Pandas, NumPy
        - **Machine Learning & Preprocessing:** Scikit-Learn, Joblib
        - **Visualization:** Matplotlib, Seaborn
        - **User Interface:** Streamlit
        - **Quality Assurance & Testing:** Pytest, NBConvert
        
        ---
        
        ### Reproduction Commands
        To reproduce the entire training, evaluation, and test pipeline locally:
        ```bash
        # 1. Install dependencies
        pip install -r requirements.txt

        # 2. Re-train models, generate evaluation reports, and serialize artifacts
        python src/train.py

        # 3. Run automated Pytest test suite
        pytest -v

        # 4. Launch the Streamlit dashboard
        streamlit run app.py
        ```
        
        ---
        
        ### Clinical & Ethical Limitations
        1. **Retrospective Single-Center Dataset:** The Cleveland cohort includes 303 patients referred for cardiac catheterization in 1989. Referral bias may inflate disease prevalence compared to community screening populations.
        2. **Absence of Modern Biomarkers:** Key clinical markers such as High-Sensitivity Troponin (hs-cTn), High-Sensitivity C-Reactive Protein (hs-CRP), B-Type Natriuretic Peptide (BNP), and detailed lipid subfractions (LDL-C, HDL-C, ApoB) are not present in this dataset.
        3. **Educational Prototype:** This platform is intended solely for educational demonstration and computer engineering portfolio evaluation. It does not provide medical diagnoses or treatment advice.
        """
    )
