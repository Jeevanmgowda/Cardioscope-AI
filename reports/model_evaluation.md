# Clinical Model Evaluation Report: Heart Disease Risk Prediction

**Evaluation Date:** October 17, 2026  
**Evaluation Dataset:** UCI Cleveland Heart Disease Held-Out Test Set ($n = 61$ patients, 28 positive, 33 negative)  
**Methodology:** Stratified 5-Fold Cross-Validation on Training Split ($n = 242$), followed by evaluation on unadulterated held-out test split.

---

## 1. Executive Summary

Coronary artery disease is a leading cause of global mortality. In clinical decision support and screening workflows, the cost of a **False Negative** (failing to identify an individual with angiographic disease, delaying critical intervention) is drastically higher than a **False Positive** (flagging a patient for non-invasive secondary testing, such as an echocardiogram or nuclear imaging).

Four machine-learning algorithms were trained, cross-validated, tuned, and evaluated under identical conditions:
1. **Logistic Regression** (Interpretable linear benchmark with $L_1/L_2$ regularization)
2. **Random Forest Classifier** (Nonlinear ensemble capturing complex physiological feature interactions)
3. **Support Vector Machine (SVM)** (Maximum-margin kernel classifier with RBF kernel)
4. **Gradient Boosting Classifier** (Iterative boosted decision trees minimizing log-loss)

**Champion Model Selected:** **Logistic Regression**  
- **Test Accuracy:** `86.89%`
- **Sensitivity / Recall:** `92.86%`
- **Specificity:** `81.82%`
- **F1-Score:** `0.8667`
- **ROC-AUC:** `0.9621`
- **Missed Cases (False Negatives):** `2` out of 28 disease cases.

---

## 2. Model Performance Benchmark Comparison

| Model | Accuracy | Recall (Sensitivity) | Specificity | Precision | F1-Score | ROC-AUC | PR-AUC | False Negatives | False Positives |
|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.8689 | 0.9286 | 0.8182 | 0.8125 | 0.8667 | 0.9621 | 0.9499 | 2 | 6 |
| **Random Forest** | 0.8689 | 0.8929 | 0.8485 | 0.8333 | 0.8621 | 0.9535 | 0.9477 | 3 | 5 |
| **Support Vector Machine** | 0.8689 | 0.9286 | 0.8182 | 0.8125 | 0.8667 | 0.9578 | 0.9374 | 2 | 6 |
| **Gradient Boosting** | 0.8852 | 0.8929 | 0.8788 | 0.8621 | 0.8772 | 0.9535 | 0.9537 | 3 | 4 |

---

## 3. Detailed Metric Definitions & Clinical Trade-offs

| Metric | Mathematical Formula | Clinical Interpretation | Clinical Priority |
|---|---|---|---|
| **Sensitivity (Recall)** | $\frac{TP}{TP + FN}$ | Proportion of patients with actual coronary artery disease correctly identified. | **Critical:** Minimizing False Negatives avoids missed diagnoses. |
| **Specificity** | $\frac{TN}{TN + FP}$ | Proportion of disease-free patients correctly classified as negative. | **High:** Minimizing unnecessary invasive catheterizations. |
| **Precision (PPV)** | $\frac{TP}{TP + FP}$ | Proportion of positive predictions that correspond to confirmed disease. | **Moderate:** Balances false alarm rates. |
| **ROC-AUC** | Area under $TPR$ vs $FPR$ curve | Discrimination capacity across all potential classification thresholds. | **High:** Assesses threshold-independent ranking ability. |
| **F1-Score** | $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$ | Harmonic mean balancing sensitivity and precision. | **High:** Primary aggregate metric for clinical balance. |

### False Negatives vs. False Positives Analysis
- **False Negatives ($FN$):** A patient with true coronary narrowing (>50%) is falsely labeled healthy. They may be discharged without cardiac workup, risking acute myocardial infarction or cardiac arrest.
- **False Positives ($FP$):** A healthy individual is flagged as potentially diseased. They undergo secondary confirmatory non-invasive testing (stress echocardiography or coronary CT angiography). While this introduces transient anxiety and testing costs, it carries no lethal health risk.
- **Decision Rationale:** Our model selection metric enforces a composite objective prioritizing high sensitivity ($Recall \ge 85\%$) and high discrimination ($ROC\text{-}AUC \ge 0.85$).

---

## 4. Diagnostic Visualizations

Generated high-resolution evaluation figures located in `reports/figures/`:
1. `model_comparison.png`: Grouped comparative performance metrics across all models.
2. `confusion_matrices.png`: 2x2 grid showing exact TP, FP, TN, FN counts for each algorithm.
3. `roc_curves.png`: Receiver Operating Characteristic curves displaying discrimination thresholds.
4. `pr_curves.png`: Precision-Recall curves displaying performance relative to disease prevalence.
5. `feature_importance.png`: Top predictive clinical indicators identified by the model.

---

## 5. Key Clinical Risk Predictors

Analysis of the champion model indicates the following primary drivers of heart disease classification:
1. **Number of Major Vessels Colored by Fluoroscopy (`ca`):** Patients with 1, 2, or 3 visible vessels display significantly elevated risk of multi-vessel coronary disease.
2. **Thallium Scintigraphy Defect (`thal`):** Reversible defects strongly correlate with inducible myocardial ischemia during physical stress.
3. **ST Depression Induced by Exercise (`oldpeak`):** Electrocardiographic depression $\ge 1.5$ mm reflects substantial subendocardial ischemia.
4. **Chest Pain Presentation (`cp`):** Asymptomatic presentation (`cp = 4`) paradoxically associates with advanced silent ischemia, whereas atypical chest pain presents lower angiographic risk.
5. **Maximum Heart Rate Achieved (`thalach`):** Inability to achieve age-predicted maximum heart rate indicates chronotropic incompetence.

---

## 6. Known Dataset & Model Limitations

1. **Sample Size & Single-Center Cohort:** The dataset contains 303 observations originally recorded at the Cleveland Clinic. External generalizability across diverse demographics requires validation on multi-center cohorts.
2. **Binary Angiographic Target:** Disease status is defined by $\ge 50\%$ lumen diameter stenosis in $\ge 1$ major vessel. It does not grade plaque vulnerability or non-obstructive coronary atherosclerosis.
3. **Missing Clinical Variables:** Essential modern cardiometabolic markers such as LDL-C, HDL-C, triglycerides, troponin, high-sensitivity C-reactive protein (hs-CRP), HbA1c, and detailed family history were unmeasured in this 1989 cohort.
4. **Educational Prototype Status:** This model is designed as an educational decision-support demonstration. It must never replace formal clinical judgment, ECG interpretation, or physician consultation.
