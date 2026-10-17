# Heart Disease Risk Prediction Using Machine Learning

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42.0-red.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Pytest-8.3.3-green.svg)](https://pytest.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

An end-to-end, reproducible, clinically inspired machine learning application that analyzes patient demographic, hemodynamic, electrocardiographic, and scintigraphic indicators to assess angiographic coronary artery disease risk.

---

> ### ⚠️ Clinical Decision Support & Educational Disclaimer
> **This application is an educational, research, and engineering decision-support prototype.** It is **NOT** a certified medical diagnostic tool or healthcare device. All risk estimates reflect mathematical patterns within the historical UCI Cleveland Clinic cohort and must **never** be used to replace formal clinical evaluation, 12-lead ECG interpretation, or medical consultation by a licensed cardiologist.

---

## 1. Project Overview & Clinical Problem Statement

Coronary artery disease (CAD) is the leading cause of mortality worldwide. Timely identification of patients who harbor significant coronary artery narrowing ($\ge 50\%$ diameter stenosis) enables targeted medical therapy, lifestyle modifications, and appropriate secondary non-invasive testing before acute myocardial infarction occurs.

This project delivers a complete engineering solution:
- **Leakage-Safe Preprocessing Pipeline:** Implements Scikit-Learn `ColumnTransformer` and `Pipeline` objects to handle continuous numerical scaling, missing value imputation, and categorical one-hot encoding fit strictly on training splits.
- **Multi-Algorithm Clinical Benchmarking:** Trains, tunes, and evaluates four classification algorithms: **Logistic Regression**, **Random Forest Classifier**, **Support Vector Machine (SVM)**, and **Gradient Boosting Classifier**.
- **Medical Evaluation Focus:** Prioritizes **Recall / Sensitivity** and **ROC-AUC** over raw accuracy to minimize dangerous **False Negatives** (failing to identify diseased individuals).
- **Interactive Health Analytics Dashboard:** A modern, restrained, responsive Streamlit web application providing individualized risk scoring, clinical factor breakdowns, and comparative diagnostic visualizations.
- **Robust Quality Assurance:** A comprehensive 19-test automated Pytest suite verifying schema integrity, biological boundary enforcement, pipeline stability, and inference reproducibility.

---

## 2. Reference Repository Audit Summary

This project draws architectural inspiration from the [Harshitraj123/Stroke-Risk-Prediction](https://github.com/Harshitraj123/Stroke-Risk-Prediction.git) repository while introducing major structural modernizations:

| Dimension | Reference Repository (Stroke) | Our Project (Heart Disease Risk Prediction) |
|---|---|---|
| **Problem Domain** | Stroke risk prediction in imbalanced cohort (4.9% positive) | Coronary artery disease angiographic classification (~46% positive) |
| **Dataset Source** | Kaggle Stroke Prediction Dataset (5,110 rows) | Gold-standard UCI Cleveland Clinic Heart Disease Dataset (303 rows) |
| **Code Architecture** | Monolithic Jupyter Notebook (`.ipynb`) + minimal data prep script | Modular Python package (`src/data_loader`, `preprocessing`, `train`, `evaluate`, `predict`) |
| **Model Persistence** | No model serialization (`.joblib` absent) | Full pipeline serialization (`models/best_model.joblib`, `models/all_models.joblib`) |
| **Inference Engine** | Absent (no standalone prediction module) | Standalone `HeartDiseasePredictor` with biological input boundary validation |
| **Interactive Frontend** | Absent (zero UI / dashboard components) | Multi-page clinical Streamlit dashboard (`app.py`) with preset profiles & risk meters |
| **Automated Testing** | Absent (no test suite) | Comprehensive 19-test Pytest suite (`tests/`) covering data, pipeline, and inference |
| **Algorithms Evaluated** | Logistic Regression, Decision Tree, Random Forest | Logistic Regression, Random Forest, SVM (RBF/Linear), Gradient Boosting |
| **Tuning & Validation** | Default parameters in notebook cells | Stratified 5-Fold Cross-Validation & Hyperparameter Tuning (`GridSearchCV`) |

---

## 3. Dataset Attribution & Feature Dictionary

- **Source:** [UCI Machine Learning Repository: Heart Disease Dataset (ID 45)](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Principal Investigators:** Dr. Robert Detrano (Cleveland Clinic Foundation & V.A. Medical Center, Long Beach), Dr. Andras Janosi, Dr. William Steinbrunn, Dr. Matthias Pfisterer.
- **Cohort Size:** 303 patient observations.
- **Target Variable (`num` & `target`):** Angiographic coronary artery status:
  - `0`: Absence of significant coronary stenosis ($< 50\%$ diameter narrowing) — **164 patients (54.1%)**
  - `1`: Presence of significant coronary stenosis ($> 50\%$ diameter narrowing) — **139 patients (45.9%)**

### Feature Dictionary

| Feature | Type | Range / Categories | Description |
|---|---|---|---|
| `age` | Integer | 29 – 77 yrs | Age of the patient in years. |
| `sex` | Binary | `0` = Female, `1` = Male | Biological sex. |
| `cp` | Categorical | `1` = Typical Angina, `2` = Atypical, `3` = Non-Anginal, `4` = Asymptomatic | Chest pain type reported upon clinical presentation. |
| `trestbps` | Integer | 94 – 200 mm Hg | Resting systolic blood pressure on hospital admission. |
| `chol` | Integer | 126 – 564 mg/dL | Serum cholesterol in mg/dL. |
| `fbs` | Binary | `0` = False ($\le 120$), `1` = True ($> 120$) | Fasting blood sugar $> 120$ mg/dL (diabetic risk indicator). |
| `restecg` | Categorical | `0` = Normal, `1` = ST-T Abnormality, `2` = LVH | Resting electrocardiographic findings. |
| `thalach` | Integer | 71 – 202 bpm | Maximum heart rate achieved during exercise treadmill test. |
| `exang` | Binary | `0` = No, `1` = Yes | Exercise-induced angina pectoris. |
| `oldpeak` | Float | 0.0 – 6.2 mm | ST-segment depression induced by exercise relative to rest. |
| `slope` | Ordinal | `1` = Upsloping, `2` = Flat, `3` = Downsloping | Slope of peak exercise ST segment. |
| `ca` | Numeric | 0.0 – 3.0 vessels | Number of major vessels (0–3) colored by catheter fluoroscopy. |
| `thal` | Categorical | `3.0` = Normal, `6.0` = Fixed, `7.0` = Reversible | Thallium-201 nuclear stress scintigraphy defect status. |

---

## 4. Project Directory Architecture

```
heart-disease-risk-prediction/
├── app.py                         # Production Streamlit clinical analytics dashboard
├── requirements.txt               # Pinned project dependencies
├── README.md                      # Primary project documentation
├── .gitignore                     # Git ignore rules for bytecode and artifacts
├── data/
│   ├── heart_disease.csv          # Standardized 303-patient dataset
│   └── README.md                  # Comprehensive data dictionary and clinical attribution
├── notebooks/
│   ├── exploratory_data_analysis.ipynb # Fully rendered Jupyter notebook with complete EDA
│   └── generate_eda_notebook.py   # Reproducible script generating the EDA notebook
├── src/
│   ├── __init__.py                # Source package initialization
│   ├── data_loader.py             # Schema validation, loading, and stratified splitting
│   ├── preprocessing.py           # Scikit-Learn ColumnTransformer and boundary validator
│   ├── train.py                   # 5-fold CV, GridSearchCV tuning, and artifact persistence
│   ├── evaluate.py                # Medical classification metrics and diagnostic plot generators
│   └── predict.py                 # Standalone inference engine with risk contributor analysis
├── models/
│   ├── best_model.joblib          # Serialized champion pipeline (Preprocessor + Classifier)
│   ├── all_models.joblib          # Serialized bundle of all 4 trained models
│   ├── model_metadata.json        # Structural metadata and CV/test evaluation metrics
│   └── README.md                  # Artifact specifications and schema instructions
├── reports/
│   ├── model_evaluation.md        # Comprehensive clinical model evaluation report
│   ├── model_comparison_results.csv # CSV table of benchmark performance
│   └── figures/                   # High-resolution PNG evaluation figures
│       ├── model_comparison.png   # Grouped comparative performance chart
│       ├── confusion_matrices.png # 2x2 confusion matrix grid
│       ├── roc_curves.png         # ROC curves across all models
│       ├── pr_curves.png          # Precision-Recall curves relative to prevalence
│       └── feature_importance.png # Top predictive standardized coefficients
└── tests/
    ├── __init__.py                # Test package initialization
    ├── test_data_validation.py    # Schema integrity, missing column, and input bound tests
    ├── test_preprocessing.py      # ColumnTransformer, imputation, and leakage tests
    └── test_prediction.py         # End-to-end inference, probability, and error tests
```

---

## 5. Machine Learning Benchmark & Results

All models were evaluated on an untouched **held-out test split** ($n = 61$, 28 positive, 33 negative) after performing **stratified 5-fold cross-validation** and hyperparameter tuning on the training split ($n = 242$).

### Model Comparison Table

| Model | Accuracy | Recall (Sensitivity) | Specificity | Precision | F1-Score | ROC-AUC | False Negatives | False Positives |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression (Champion)** | **86.89%** | **92.86%** | **81.82%** | **81.25%** | **0.8667** | **0.9621** | **2** | **6** |
| **Support Vector Machine (SVM)** | 86.89% | 92.86% | 81.82% | 81.25% | 0.8667 | 0.9578 | 2 | 6 |
| **Random Forest Classifier** | 86.89% | 89.29% | 84.85% | 83.33% | 0.8621 | 0.9535 | 3 | 5 |
| **Gradient Boosting Classifier** | 88.52% | 89.29% | 87.88% | 86.21% | 0.8772 | 0.9535 | 3 | 4 |

### Why Logistic Regression is the Clinical Champion:
1. **Minimizes Life-Threatening False Negatives:** Missed only **2** out of 28 confirmed diseased patients (**92.86% Sensitivity**).
2. **Superior Discrimination:** Highest test **ROC-AUC (0.9621)**, demonstrating optimal risk ranking across all clinical thresholds.
3. **Medical Transparency:** Avoids opaque black-box opacity; coefficients directly align with cardiovascular pathology.

---

## 6. Installation & Execution Guide

### Step 1: Clone Repository & Set Up Virtual Environment
```bash
git clone https://github.com/your-username/heart-disease-risk-prediction.git
cd heart-disease-risk-prediction

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Machine Learning Training Pipeline
Retrains all models, runs 5-fold cross validation, executes hyperparameter grid searches, serializes models to `models/`, and produces evaluation charts in `reports/figures/`:
```bash
python src/train.py
```

### Step 4: Execute Automated Test Suite
Runs the 19 automated Pytest unit and integration tests:
```bash
pytest -v
```

### Step 5: Launch Streamlit Health Analytics Dashboard
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 7. Streamlit Dashboard Features

The application is structured into five cohesive sections:
1. **Executive Dashboard:** High-level project KPIs, validation cohort metrics, champion model badge, and end-to-end workflow diagram.
2. **Patient Risk Assessment:** Structured 3-column input form (Demographics, Cardiac Symptoms, ECG & Fluoroscopy) with clinical range validation, preset profile selectors (Healthy, Borderline, High-Risk), visual risk meter gauge, and individualized risk factor breakdowns.
3. **Model Benchmark & Evaluation:** Interactive model switcher, comprehensive metrics table, 2x2 confusion matrix grid, ROC curves, Precision-Recall curves, and false-negative vs false-positive trade-off analysis.
4. **Dataset Insights & EDA:** Interactive distributions of age, cholesterol, and resting blood pressure by diagnosis, exercise ST depression boxplots, fluoroscopy vessel impact, and correlation heatmaps.
5. **Methodology & Clinical Limitations:** Formal technical documentation, metric definitions, reproducibility commands, and ethical limitations.

---

## 8. Automated Testing & Verification Summary

The test suite in `tests/` verifies every aspect of data integrity and model reliability:
```bash
tests/test_data_validation.py::test_dataset_file_exists PASSED           [  5%]
tests/test_data_validation.py::test_load_heart_disease_data_schema PASSED [ 10%]
tests/test_data_validation.py::test_validate_raw_dataframe_missing_column PASSED [ 15%]
tests/test_data_validation.py::test_validate_raw_dataframe_empty PASSED  [ 21%]
tests/test_data_validation.py::test_prepare_features_and_target PASSED   [ 26%]
tests/test_data_validation.py::test_stratified_split_preserves_class_balance PASSED [ 31%]
tests/test_data_validation.py::test_validate_input_record_valid PASSED   [ 36%]
tests/test_data_validation.py::test_validate_input_record_missing_keys PASSED [ 42%]
tests/test_data_validation.py::test_validate_input_record_out_of_bounds PASSED [ 47%]
tests/test_prediction.py::test_predictor_initialization PASSED           [ 52%]
tests/test_prediction.py::test_predict_patient_healthy PASSED            [ 57%]
tests/test_prediction.py::test_predict_patient_high_risk PASSED          [ 63%]
tests/test_prediction.py::test_feature_ordering_enforcement PASSED       [ 68%]
tests/test_prediction.py::test_invalid_input_raises_value_error PASSED   [ 73%]
tests/test_prediction.py::test_missing_model_file_raises_error PASSED    [ 78%]
tests/test_preprocessing.py::test_build_preprocessor_structure PASSED    [ 84%]
tests/test_preprocessing.py::test_preprocessor_handles_missing_values PASSED [ 89%]
tests/test_preprocessing.py::test_get_feature_names_out PASSED           [ 94%]
tests/test_preprocessing.py::test_leakage_safety_train_test_independence PASSED [100%]

============================= 19 passed in 1.96s ==============================
```

---

## 9. Limitations & Clinical Considerations

1. **Cohort Demographics:** The 303 patients were referred for catheterization at the Cleveland Clinic in 1989. Referral bias may result in higher baseline disease prevalence than observed in routine primary care screening.
2. **Binary Angiographic Threshold:** The classification is defined as $\ge 50\%$ vessel narrowing in at least one major coronary artery. It does not measure non-obstructive plaque burden or vulnerable lipid pools.
3. **Unmeasured Clinical Biomarkers:** Modern cardiovascular markers such as High-Sensitivity Cardiac Troponin (hs-cTn), High-Sensitivity C-Reactive Protein (hs-CRP), B-Type Natriuretic Peptide (BNP), and detailed lipid subfractions (ApoB, Lp(a)) were unmeasured in this historical cohort.
4. **Educational Decision-Support Prototype:** The software is designed to demonstrate computer science and machine-learning engineering best practices. It must not be deployed as an autonomous diagnostic device.

---

## 10. License & Citations

- **Dataset License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Original Citation:**
  > Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Gopal, R., Alemayehu, S., & Froelicher, V. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* American Journal of Cardiology, 64(5), 304–310.
