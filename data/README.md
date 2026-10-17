# UCI Heart Disease Dataset Documentation

## 1. Overview and Attribution

- **Dataset Name:** Heart Disease Dataset (Cleveland Clinic Foundation)
- **Primary Citation:** Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Gopal, R., Alemayehu, S., & Froelicher, V. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* American Journal of Cardiology, 64(5), 304–310.
- **Repository URL:** [UCI Machine Learning Repository: Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Donor / Creators:**
  1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
  2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
  3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
  4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
- **License / Terms:** Creative Commons Attribution 4.0 International (CC BY 4.0). Open for academic, research, and educational use with attribution.

---

## 2. Dataset Dimensions and Target Definition

- **Total Observations:** 303 patient records
- **Total Raw Columns:** 14 attributes + 1 derived binary classification target (`target`)
- **Target Variable (`num`):** Angiographic coronary artery disease status (narrowing of major vessels):
  - `0`: `< 50%` diameter narrowing (Absence of significant angiographic heart disease)
  - `1, 2, 3, 4`: `> 50%` diameter narrowing in 1, 2, 3, or 4 major coronary arteries
- **Binary Target (`target`):**
  - `0`: Absence of disease (`num == 0`) — **164 patients (54.1%)**
  - `1`: Presence of disease (`num > 0`) — **139 patients (45.9%)**

> Note: This dataset provides an angiographic classification based on objective cardiac catheterization findings. Predictions from models trained on this dataset indicate angiographic disease presence rather than a calibrated, longitudinal prospective risk probability.

---

## 3. Feature Dictionary

| Feature | Data Type | Range / Categories | Description and Clinical Significance |
|---|---|---|---|
| `age` | Integer | 29 – 77 years | Age of the patient in completed years. Risk of cardiovascular disease increases progressively with age. |
| `sex` | Integer (Binary) | `0` = Female, `1` = Male | Biological sex of the patient. Males historically present coronary artery disease at earlier ages. |
| `cp` | Integer (Nominal) | `1` = Typical Angina<br>`2` = Atypical Angina<br>`3` = Non-Anginal Pain<br>`4` = Asymptomatic | Chest pain type reported by patient upon clinical presentation. Note: Asymptomatic (`4`) often associates strongly with silent ischemia in severe disease. |
| `trestbps` | Integer | 94 – 200 mm Hg | Resting systolic blood pressure on admission to the hospital. Clinical hypertension is defined as systolic BP ≥ 130 mm Hg. |
| `chol` | Integer | 126 – 564 mg/dL | Serum cholesterol in mg/dL. Elevated serum cholesterol contributes to coronary atheroma plaque formation. |
| `fbs` | Integer (Binary) | `0` = False (≤ 120 mg/dL)<br>`1` = True (> 120 mg/dL) | Fasting blood sugar > 120 mg/dL, an indicator of diabetes mellitus or impaired glucose tolerance. |
| `restecg` | Integer (Nominal) | `0` = Normal<br>`1` = ST-T wave abnormality<br>`2` = Left ventricular hypertrophy | Resting electrocardiographic findings according to Estes' criteria or presence of T-wave inversion / ST changes. |
| `thalach` | Integer | 71 – 202 bpm | Maximum heart rate achieved during symptom-limited treadmill exercise stress testing. |
| `exang` | Integer (Binary) | `0` = No, `1` = Yes | Exercise-induced angina pectoris during treadmill stress testing. |
| `oldpeak` | Float | 0.0 – 6.2 mm | ST-segment depression induced by exercise relative to resting baseline on ECG (indicates myocardial ischemia). |
| `slope` | Integer (Ordinal) | `1` = Upsloping<br>`2` = Flat<br>`3` = Downsloping | The slope of the peak exercise ST-segment. Flat or downsloping profiles correlate with severe ischemia. |
| `ca` | Float | 0.0 – 3.0 (4 missing) | Number of major coronary vessels (0–3) colored by fluoroscopy during catheterization. |
| `thal` | Float | `3.0` = Normal<br>`6.0` = Fixed Defect<br>`7.0` = Reversible Defect (2 missing) | Thallium-201 nuclear stress scintigraphy defect status. Reversible defect indicates inducible myocardial ischemia. |

---

## 4. Missing Values & Preprocessing Strategy

- `ca` has 4 missing records (`1.32%`). Handled via median imputation computed strictly on the training partition.
- `thal` has 2 missing records (`0.66%`). Handled via most-frequent (mode) imputation computed strictly on the training partition.
- Categorical features (`cp`, `restecg`, `slope`, `thal`) are one-hot encoded using Scikit-Learn `ColumnTransformer`.
- Continuous numerical features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) are scaled using `StandardScaler`.
- Binary features (`sex`, `fbs`, `exang`) and ordinal `ca` are passed through cleanly.
- All transformations are encapsulated in an sklearn `Pipeline` fit exclusively on training data to prevent data leakage.
