# Serialized Model Artifacts & Metadata

This directory stores the serialized machine-learning models, preprocessors, and metadata produced by `src/train.py`.

## Artifact Inventory

- **`best_model.joblib`**: The designated champion Scikit-Learn `Pipeline`. Contains the complete fitted `ColumnTransformer` (scaling, imputation, one-hot encoding) combined with the optimized classifier (Logistic Regression).
- **`all_models.joblib`**: A Python dictionary containing all four fitted, tuned pipelines (`Logistic Regression`, `Random Forest`, `Support Vector Machine`, `Gradient Boosting`).
- **`model_metadata.json`**: Structural metadata detailing dataset split statistics, hyperparameter settings, 5-fold cross-validation metrics, held-out test evaluation scores, and clinical feature definitions.

## Champion Model Specifications

- **Algorithm:** Logistic Regression ($L_2$ Regularized, `C=0.5`, `solver='liblinear'`)
- **Held-Out Test ROC-AUC:** `0.9621`
- **Sensitivity / Recall:** `92.86%` (26/28 positive disease cases detected)
- **Specificity:** `81.82%` (27/33 negative cases correctly confirmed)
- **Overall Accuracy:** `86.89%`
- **Serialization Framework:** Joblib

## Reproducibility

To re-train all models and regenerate these serialized artifacts from scratch, execute:

```bash
python src/train.py
```
