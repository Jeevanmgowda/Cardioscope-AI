"""Model evaluation and visualization module for Heart Disease Risk Prediction.

Calculates medical classification metrics (accuracy, precision, recall/sensitivity,
specificity, F1-score, ROC-AUC, PR-AUC), plots diagnostic curves, and generates
comprehensive clinical evaluation reports.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Styling configuration for consistent medical publication-quality charts
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PALETTE = {
    "primary": "#0284c7",    # Clinical Cyan/Blue
    "secondary": "#0f766e",  # Deep Teal
    "danger": "#e11d48",     # Coral / Alert Red
    "warning": "#d97706",    # Amber
    "success": "#16a34a",    # Forest Green
    "dark": "#1e293b",       # Slate Dark
    "light": "#f8fafc",      # Slate Light
}


def compute_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_prob: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Dict[str, float]:
    """Compute standard and clinical binary classification metrics.

    Calculates:
    - Accuracy
    - Precision (Positive Predictive Value)
    - Recall (Sensitivity / True Positive Rate)
    - Specificity (True Negative Rate)
    - F1-Score
    - ROC-AUC
    - PR-AUC (Average Precision)

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Model binary predicted class (0 or 1).
        y_prob: Predicted probability for class 1 (optional).

    Returns:
        Dict[str, float]: Computed metrics formatted to 4 decimal places.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc = float(roc_auc_score(y_true, y_prob)) if y_prob is not None else 0.0
    pr_auc = float(average_precision_score(y_true, y_prob)) if y_prob is not None else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def plot_model_comparison_bars(
    comparison_df: pd.DataFrame,
    save_path: str,
) -> None:
    """Generate a grouped bar chart comparing performance metrics across models.

    Args:
        comparison_df: DataFrame with models as rows and metrics as columns.
        save_path: Destination PNG filepath.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metrics_to_plot = ["accuracy", "recall", "precision", "specificity", "f1_score", "roc_auc"]
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]

    df_melted = comparison_df.reset_index().melt(
        id_vars="model",
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    palette = sns.color_palette("mako", n_colors=len(available_metrics))

    sns.barplot(
        data=df_melted,
        x="model",
        y="Score",
        hue="Metric",
        palette=palette,
        ax=ax,
        edgecolor="#ffffff",
        linewidth=0.8,
    )

    ax.set_title("Clinical Model Benchmark Comparison (Held-Out Test Set)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Machine Learning Algorithm", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11, fontweight="semibold")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0.1:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#334155",
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices_grid(
    models_dict: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: str,
) -> None:
    """Plot confusion matrices for all evaluated models in a 2x2 grid.

    Args:
        models_dict: Dict mapping model names to fitted pipelines.
        X_test: Test features.
        y_test: True test labels.
        save_path: Destination PNG filepath.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_models = len(models_dict)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), dpi=300)
    axes = axes.flatten()

    for idx, (name, pipeline) in enumerate(models_dict.items()):
        if idx >= 4:
            break
        ax = axes[idx]
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Absence (0)", "Presence (1)"],
            yticklabels=["Absence (0)", "Presence (1)"],
            annot_kws={"size": 13, "weight": "bold"},
        )

        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        ax.set_title(f"{name}\nSensitivity: {recall:.1%} | Specificity: {specificity:.1%}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted Diagnosis", fontsize=9.5)
        ax.set_ylabel("True Angiographic Status", fontsize=9.5)

    plt.suptitle("Confusion Matrix Grid — Clinical Classification Performance", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_roc_and_pr_curves(
    models_dict: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    roc_save_path: str,
    pr_save_path: str,
) -> None:
    """Generate both ROC and Precision-Recall diagnostic curves for all models.

    Args:
        models_dict: Dict of fitted model pipelines.
        X_test: Test features.
        y_test: True test labels.
        roc_save_path: Destination path for ROC curve.
        pr_save_path: Destination path for PR curve.
    """
    os.makedirs(os.path.dirname(roc_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(pr_save_path), exist_ok=True)

    # 1. ROC Curve
    fig_roc, ax_roc = plt.subplots(figsize=(7.5, 6), dpi=300)
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["danger"], PALETTE["warning"]]

    for idx, (name, pipeline) in enumerate(models_dict.items()):
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            score = roc_auc_score(y_test, y_prob)
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {score:.3f})", color=colors[idx % len(colors)], linewidth=2)

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.500)")
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.05])
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="semibold")
    ax_roc.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="semibold")
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold", pad=12)
    ax_roc.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig_roc.savefig(roc_save_path, bbox_inches="tight")
    plt.close(fig_roc)

    # 2. Precision-Recall Curve
    fig_pr, ax_pr = plt.subplots(figsize=(7.5, 6), dpi=300)
    baseline_prevalence = float(y_test.sum() / len(y_test))

    for idx, (name, pipeline) in enumerate(models_dict.items()):
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            ax_pr.plot(recall, precision, label=f"{name} (AP = {pr_auc:.3f})", color=colors[idx % len(colors)], linewidth=2)

    ax_pr.axhline(y=baseline_prevalence, color="k", linestyle="--", alpha=0.5, label=f"Prevalence Baseline ({baseline_prevalence:.1%})")
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight="semibold")
    ax_pr.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="semibold")
    ax_pr.set_title("Precision-Recall (PR) Curves", fontsize=13, fontweight="bold", pad=12)
    ax_pr.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    fig_pr.savefig(pr_save_path, bbox_inches="tight")
    plt.close(fig_pr)


def plot_feature_importance_ranking(
    pipeline: object,
    save_path: str,
    top_n: int = 15,
) -> None:
    """Plot feature importances or logistic regression odds/coefficients.

    Args:
        pipeline: Fitted scikit-learn pipeline with preprocessor and classifier.
        save_path: Destination PNG filepath.
        top_n: Number of top features to plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    preprocessor = pipeline.named_steps.get("preprocessor")
    classifier = pipeline.named_steps.get("classifier")

    if not preprocessor or not classifier:
        return

    # Extract feature names
    feature_names = preprocessor.get_feature_names_out()

    importances = None
    title = "Top Feature Importances"
    xlabel = "Relative Importance"

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        title = f"Top {top_n} Predictive Features ({type(classifier).__name__})"
        xlabel = "Gini Feature Importance"
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
        title = f"Top {top_n} Standardized Coefficients ({type(classifier).__name__})"
        xlabel = "Absolute Standardized Coefficient Magnitude"

    if importances is None:
        return

    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values(by="Importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
    sns.barplot(
        data=feat_df,
        y="Feature",
        x="Importance",
        palette="viridis",
        ax=ax,
        edgecolor="#ffffff",
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=10, fontweight="semibold")
    ax.set_ylabel("Clinical / Encoded Feature", fontsize=10, fontweight="semibold")

    for p in ax.patches:
        width = p.get_width()
        ax.annotate(
            f"{width:.3f}",
            (width, p.get_y() + p.get_height() / 2.0),
            ha="left",
            va="center",
            fontsize=8,
            color="#334155",
            xytext=(4, 0),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
