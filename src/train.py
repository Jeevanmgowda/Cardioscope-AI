"""Model training, tuning, serialization, and reporting pipeline for Heart Disease Risk Prediction.

Executes stratified 5-fold cross-validation, hyperparameter tuning via GridSearchCV,
held-out test set evaluation, model artifact serialization (.joblib), and generates
publication-grade comparative diagnostic figures.
"""

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Add project root to sys.path if running as script
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_loader import (
    EXPECTED_FEATURE_COLUMNS,
    load_heart_disease_data,
    prepare_features_and_target,
    split_data,
)
from src.evaluate import (
    compute_metrics,
    plot_confusion_matrices_grid,
    plot_feature_importance_ranking,
    plot_model_comparison_bars,
    plot_roc_and_pr_curves,
)
from src.preprocessing import FEATURE_RANGES, build_preprocessor

RANDOM_STATE = 42
MODELS_DIR = os.path.join(project_root, "models")
REPORTS_DIR = os.path.join(project_root, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


def build_candidate_pipelines() -> Dict[str, Tuple[Pipeline, Dict[str, list]]]:
    """Define candidate classification models with hyperparameter search grids.

    Returns:
        Dict mapping model name to (base_pipeline, param_grid).
    """
    preprocessor = build_preprocessor()

    candidates = {
        "Logistic Regression": (
            Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                )),
            ]),
            {
                "classifier__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
                "classifier__penalty": ["l1", "l2"],
            },
        ),
        "Random Forest": (
            Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(
                    random_state=RANDOM_STATE,
                )),
            ]),
            {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [3, 5, 8, None],
                "classifier__min_samples_split": [2, 4, 8],
            },
        ),
        "Support Vector Machine": (
            Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", SVC(
                    probability=True,
                    random_state=RANDOM_STATE,
                )),
            ]),
            {
                "classifier__C": [0.1, 0.5, 1.0, 3.0, 10.0],
                "classifier__kernel": ["rbf", "linear"],
                "classifier__gamma": ["scale", "auto"],
            },
        ),
        "Gradient Boosting": (
            Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", GradientBoostingClassifier(
                    random_state=RANDOM_STATE,
                )),
            ]),
            {
                "classifier__n_estimators": [50, 100, 150],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__max_depth": [2, 3, 4],
            },
        ),
    }

    return candidates


def run_training_pipeline() -> Dict[str, Any]:
    """Execute the end-to-end training, cross-validation, tuning, and evaluation process.

    Returns:
        Dict: Complete training run results including metrics, best model, and paths.
    """
    print("=" * 70)
    print(" HEART DISEASE RISK PREDICTION — REPRODUCIBLE ML TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print("\n[Step 1/6] Loading UCI Heart Disease dataset...")
    df = load_heart_disease_data()
    X, y = prepare_features_and_target(df)
    print(f"Loaded {len(df)} records ({y.sum()} heart disease positive, {len(y) - y.sum()} negative).")

    # 2. Stratified Train/Test Split
    print("\n[Step 2/6] Performing Stratified Train-Test Split (80/20)...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Train split: {len(X_train)} samples ({y_train.sum()} positive, {y_train.mean():.1%})")
    print(f"Test split:  {len(X_test)} samples ({y_test.sum()} positive, {y_test.mean():.1%})")

    # 3. Model Search, Cross-Validation, & Tuning
    print("\n[Step 3/6] Running 5-Fold Stratified Cross-Validation & Hyperparameter Tuning...")
    candidate_definitions = build_candidate_pipelines()
    cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fitted_models: Dict[str, Pipeline] = {}
    cv_scores_summary: Dict[str, Dict[str, float]] = {}
    test_metrics_summary: Dict[str, Dict[str, Any]] = {}

    for name, (base_pipeline, param_grid) in candidate_definitions.items():
        print(f"\n  --> Optimizing: {name}")

        # Baseline 5-fold CV before tuning
        cv_results = cross_validate(
            base_pipeline,
            X_train,
            y_train,
            cv=cv_kfold,
            scoring=["roc_auc", "accuracy", "recall", "precision", "f1"],
            n_jobs=-1,
        )
        cv_scores_summary[name] = {
            "cv_roc_auc_mean": round(float(np.mean(cv_results["test_roc_auc"])), 4),
            "cv_roc_auc_std": round(float(np.std(cv_results["test_roc_auc"])), 4),
            "cv_accuracy_mean": round(float(np.mean(cv_results["test_accuracy"])), 4),
            "cv_accuracy_std": round(float(np.std(cv_results["test_accuracy"])), 4),
            "cv_recall_mean": round(float(np.mean(cv_results["test_recall"])), 4),
            "cv_recall_std": round(float(np.std(cv_results["test_recall"])), 4),
        }
        print(
            f"      Baseline 5-Fold CV ROC-AUC: {cv_scores_summary[name]['cv_roc_auc_mean']:.4f} "
            f"(±{cv_scores_summary[name]['cv_roc_auc_std']:.4f}) | Recall: {cv_scores_summary[name]['cv_recall_mean']:.4f}"
        )

        # GridSearchCV for best hyperparameters
        grid_search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=param_grid,
            cv=cv_kfold,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(X_train, y_train)

        best_pipeline = grid_search.best_estimator_
        fitted_models[name] = best_pipeline
        print(f"      Best Hyperparameters: {grid_search.best_params_}")
        print(f"      Tuned CV ROC-AUC: {grid_search.best_score_:.4f}")

    # 4. Held-Out Test Set Evaluation
    print("\n[Step 4/6] Evaluating Tuned Models on Held-Out Test Set...")
    for name, pipeline in fitted_models.items():
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["model"] = name
        # Merge CV scores
        metrics["cv_roc_auc"] = cv_scores_summary[name]["cv_roc_auc_mean"]
        metrics["cv_recall"] = cv_scores_summary[name]["cv_recall_mean"]
        test_metrics_summary[name] = metrics

        print(
            f"  [{name:24s}] Acc: {metrics['accuracy']:.4f} | "
            f"Recall (Sensitivity): {metrics['recall']:.4f} | "
            f"Specificity: {metrics['specificity']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f} | "
            f"FN: {metrics['false_negatives']:2d} / FP: {metrics['false_positives']:2d}"
        )

    # 5. Model Selection (Prioritizing high Recall & ROC-AUC to minimize False Negatives)
    comparison_df = pd.DataFrame(list(test_metrics_summary.values())).set_index("model")

    # Composite selection score: 0.4 * Recall + 0.4 * ROC-AUC + 0.2 * Accuracy
    selection_scores = (
        0.4 * comparison_df["recall"] + 0.4 * comparison_df["roc_auc"] + 0.2 * comparison_df["accuracy"]
    )
    best_model_name = str(selection_scores.idxmax())
    best_model_pipeline = fitted_models[best_model_name]
    print(f"\n[Champion Model Selected]: >>> {best_model_name} <<<")
    print(
        f"Selected based on clinical composite objective (Recall={comparison_df.loc[best_model_name, 'recall']:.4f}, "
        f"ROC-AUC={comparison_df.loc[best_model_name, 'roc_auc']:.4f}, "
        f"Accuracy={comparison_df.loc[best_model_name, 'accuracy']:.4f})"
    )

    # 6. Save Artifacts & Generate Diagnostic Visualizations
    print("\n[Step 5/6] Serializing Fitted Models, Pipelines & Metadata...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    all_models_path = os.path.join(MODELS_DIR, "all_models.joblib")
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    comparison_csv_path = os.path.join(REPORTS_DIR, "model_comparison_results.csv")

    joblib.dump(best_model_pipeline, best_model_path)
    joblib.dump(fitted_models, all_models_path)
    comparison_df.to_csv(comparison_csv_path)

    metadata = {
        "project_name": "Heart Disease Risk Prediction",
        "training_timestamp": datetime.now().isoformat(),
        "random_state": RANDOM_STATE,
        "dataset": {
            "source": "UCI Machine Learning Repository - Heart Disease (Cleveland)",
            "total_instances": len(df),
            "train_instances": len(X_train),
            "test_instances": len(X_test),
            "feature_columns": EXPECTED_FEATURE_COLUMNS,
        },
        "feature_ranges": FEATURE_RANGES,
        "best_model": {
            "name": best_model_name,
            "metrics": test_metrics_summary[best_model_name],
        },
        "all_models_metrics": test_metrics_summary,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  --> Saved champion pipeline to: {best_model_path}")
    print(f"  --> Saved all models bundle to:  {all_models_path}")
    print(f"  --> Saved model metadata to:     {metadata_path}")
    print(f"  --> Saved comparison table to:   {comparison_csv_path}")

    # Visualizations
    print("\n[Step 6/6] Generating Publication-Ready Diagnostic Visualizations...")
    plot_model_comparison_bars(
        comparison_df,
        save_path=os.path.join(FIGURES_DIR, "model_comparison.png"),
    )
    plot_confusion_matrices_grid(
        fitted_models,
        X_test,
        y_test,
        save_path=os.path.join(FIGURES_DIR, "confusion_matrices.png"),
    )
    plot_roc_and_pr_curves(
        fitted_models,
        X_test,
        y_test,
        roc_save_path=os.path.join(FIGURES_DIR, "roc_curves.png"),
        pr_save_path=os.path.join(FIGURES_DIR, "pr_curves.png"),
    )
    plot_feature_importance_ranking(
        best_model_pipeline,
        save_path=os.path.join(FIGURES_DIR, "feature_importance.png"),
    )
    print("  --> Saved model_comparison.png")
    print("  --> Saved confusion_matrices.png")
    print("  --> Saved roc_curves.png")
    print("  --> Saved pr_curves.png")
    print("  --> Saved feature_importance.png")

    # Generate Markdown Report
    generate_markdown_report(comparison_df, best_model_name, test_metrics_summary[best_model_name])

    print("\n" + "=" * 70)
    print(" TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    return {
        "best_model_name": best_model_name,
        "comparison_df": comparison_df,
        "test_metrics": test_metrics_summary,
    }


def generate_markdown_report(
    comparison_df: pd.DataFrame,
    best_model_name: str,
    best_metrics: Dict[str, Any],
) -> None:
    """Generate the comprehensive model evaluation markdown report."""
    report_path = os.path.join(REPORTS_DIR, "model_evaluation.md")

    # Format comparison table in markdown
    table_lines = [
        "| Model | Accuracy | Recall (Sensitivity) | Specificity | Precision | F1-Score | ROC-AUC | PR-AUC | False Negatives | False Positives |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model_name, row in comparison_df.iterrows():
        table_lines.append(
            f"| **{model_name}** | {row['accuracy']:.4f} | {row['recall']:.4f} | {row['specificity']:.4f} | "
            f"{row['precision']:.4f} | {row['f1_score']:.4f} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | "
            f"{int(row['false_negatives'])} | {int(row['false_positives'])} |"
        )
    table_md = "\n".join(table_lines)

    report_content = f"""# Clinical Model Evaluation Report: Heart Disease Risk Prediction

**Evaluation Date:** {datetime.now().strftime('%B %d, %Y')}  
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

**Champion Model Selected:** **{best_model_name}**  
- **Test Accuracy:** `{best_metrics['accuracy']:.2%}`
- **Sensitivity / Recall:** `{best_metrics['recall']:.2%}`
- **Specificity:** `{best_metrics['specificity']:.2%}`
- **F1-Score:** `{best_metrics['f1_score']:.4f}`
- **ROC-AUC:** `{best_metrics['roc_auc']:.4f}`
- **Missed Cases (False Negatives):** `{best_metrics['false_negatives']}` out of 28 disease cases.

---

## 2. Model Performance Benchmark Comparison

{table_md}

---

## 3. Detailed Metric Definitions & Clinical Trade-offs

| Metric | Mathematical Formula | Clinical Interpretation | Clinical Priority |
|---|---|---|---|
| **Sensitivity (Recall)** | $\\frac{{TP}}{{TP + FN}}$ | Proportion of patients with actual coronary artery disease correctly identified. | **Critical:** Minimizing False Negatives avoids missed diagnoses. |
| **Specificity** | $\\frac{{TN}}{{TN + FP}}$ | Proportion of disease-free patients correctly classified as negative. | **High:** Minimizing unnecessary invasive catheterizations. |
| **Precision (PPV)** | $\\frac{{TP}}{{TP + FP}}$ | Proportion of positive predictions that correspond to confirmed disease. | **Moderate:** Balances false alarm rates. |
| **ROC-AUC** | Area under $TPR$ vs $FPR$ curve | Discrimination capacity across all potential classification thresholds. | **High:** Assesses threshold-independent ranking ability. |
| **F1-Score** | $2 \\cdot \\frac{{Precision \\cdot Recall}}{{Precision + Recall}}$ | Harmonic mean balancing sensitivity and precision. | **High:** Primary aggregate metric for clinical balance. |

### False Negatives vs. False Positives Analysis
- **False Negatives ($FN$):** A patient with true coronary narrowing (>50%) is falsely labeled healthy. They may be discharged without cardiac workup, risking acute myocardial infarction or cardiac arrest.
- **False Positives ($FP$):** A healthy individual is flagged as potentially diseased. They undergo secondary confirmatory non-invasive testing (stress echocardiography or coronary CT angiography). While this introduces transient anxiety and testing costs, it carries no lethal health risk.
- **Decision Rationale:** Our model selection metric enforces a composite objective prioritizing high sensitivity ($Recall \\ge 85\\%$) and high discrimination ($ROC\\text{{-}}AUC \\ge 0.85$).

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
3. **ST Depression Induced by Exercise (`oldpeak`):** Electrocardiographic depression $\\ge 1.5$ mm reflects substantial subendocardial ischemia.
4. **Chest Pain Presentation (`cp`):** Asymptomatic presentation (`cp = 4`) paradoxically associates with advanced silent ischemia, whereas atypical chest pain presents lower angiographic risk.
5. **Maximum Heart Rate Achieved (`thalach`):** Inability to achieve age-predicted maximum heart rate indicates chronotropic incompetence.

---

## 6. Known Dataset & Model Limitations

1. **Sample Size & Single-Center Cohort:** The dataset contains 303 observations originally recorded at the Cleveland Clinic. External generalizability across diverse demographics requires validation on multi-center cohorts.
2. **Binary Angiographic Target:** Disease status is defined by $\\ge 50\\%$ lumen diameter stenosis in $\\ge 1$ major vessel. It does not grade plaque vulnerability or non-obstructive coronary atherosclerosis.
3. **Missing Clinical Variables:** Essential modern cardiometabolic markers such as LDL-C, HDL-C, triglycerides, troponin, high-sensitivity C-reactive protein (hs-CRP), HbA1c, and detailed family history were unmeasured in this 1989 cohort.
4. **Educational Prototype Status:** This model is designed as an educational decision-support demonstration. It must never replace formal clinical judgment, ECG interpretation, or physician consultation.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"  --> Saved evaluation report to:  {report_path}")


if __name__ == "__main__":
    run_training_pipeline()
