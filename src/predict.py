"""Inference and risk scoring engine for Heart Disease Risk Prediction.

Provides the HeartDiseasePredictor class for loading serialized model pipelines,
validating patient inputs, calculating calibrated risk scores, and generating
interpretable risk factor breakdowns with educational disclaimers.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path if running as script
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_loader import EXPECTED_FEATURE_COLUMNS
from src.preprocessing import FEATURE_RANGES, validate_input_record

DEFAULT_MODEL_PATH = os.path.join(project_root, "models", "best_model.joblib")
DEFAULT_METADATA_PATH = os.path.join(project_root, "models", "model_metadata.json")

CLINICAL_DISCLAIMER = (
    "DISCLAIMER: This application is an educational and academic decision-support prototype "
    "developed for research demonstration. It is NOT a certified medical diagnostic device or "
    "clinical decision tool. Predictions indicate mathematical likelihood based on patterns in "
    "historical training data (UCI Heart Disease cohort) and must never replace formal clinical "
    "assessment, 12-lead ECG interpretation, or medical advice by a licensed healthcare provider."
)


class HeartDiseasePredictor:
    """Production-grade inference engine for Heart Disease Risk Prediction."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ):
        """Initialize the predictor by loading serialized pipeline and metadata.

        Args:
            model_path: Optional path to serialized joblib pipeline.
            metadata_path: Optional path to JSON metadata file.
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.metadata_path = metadata_path or DEFAULT_METADATA_PATH
        self.pipeline = None
        self.metadata = {}
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load serialized model pipeline and metadata from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model artifact not found at '{self.model_path}'. "
                "Please train the model first by running: `python src/train.py`"
            )

        try:
            self.pipeline = joblib.load(self.model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to deserialize model artifact at '{self.model_path}': {exc}"
            ) from exc

        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}

    def predict_patient(
        self,
        patient_data: Dict[str, Union[int, float]],
    ) -> Dict[str, Any]:
        """Perform clinical inference on a single patient record.

        Args:
            patient_data: Dictionary mapping all 13 required feature keys to values.

        Returns:
            Dict containing prediction, estimated probability, risk category,
            risk factor breakdown, and educational disclaimer.

        Raises:
            ValueError: If input fails validation against physiological ranges.
            RuntimeError: If model pipeline is not loaded.
        """
        if self.pipeline is None:
            raise RuntimeError("Inference pipeline is not initialized.")

        # 1. Validate input schema and ranges
        is_valid, validation_errors = validate_input_record(patient_data)
        if not is_valid:
            raise ValueError(f"Patient input validation failed: {'; '.join(validation_errors)}")

        # 2. Construct ordered single-row DataFrame enforcing strict column order
        input_df = pd.DataFrame([patient_data])[EXPECTED_FEATURE_COLUMNS]

        # 3. Generate prediction and probability
        raw_pred = int(self.pipeline.predict(input_df)[0])
        probability = None
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(input_df)[0]
            probability = float(probs[1])
        else:
            probability = float(raw_pred)

        # 4. Determine clinical risk category
        if probability < 0.35:
            risk_category = "Low Clinical Likelihood"
            risk_color = "#16a34a"  # Green
            interpretation = (
                "The model predicts an absence of significant coronary artery disease "
                "(< 50% angiographic diameter stenosis). Patient indicators align predominantly "
                "with non-diseased baseline profiles."
            )
        elif 0.35 <= probability < 0.65:
            risk_category = "Moderate / Borderline Clinical Likelihood"
            risk_color = "#d97706"  # Amber
            interpretation = (
                "The patient presents borderline cardiovascular parameters. Further non-invasive "
                "evaluation (e.g., stress echocardiography or CT coronary angiography) may be warranted."
            )
        else:
            risk_category = "Elevated Clinical Likelihood"
            risk_color = "#e11d48"  # Red
            interpretation = (
                "The model predicts the presence of significant coronary artery disease "
                "(> 50% angiographic diameter stenosis). Clinical cardiology referral and "
                "investigation are recommended."
            )

        # 5. Extract individualized contributing risk factors
        contributing_factors = self._analyze_risk_factors(patient_data)

        # 6. Assemble clinical response
        model_name = self.metadata.get("best_model", {}).get("name", "Best Trained Model")

        return {
            "prediction_class": raw_pred,
            "prediction_label": "Presence of Disease (Positive)" if raw_pred == 1 else "Absence of Disease (Negative)",
            "risk_score": round(probability, 4),
            "risk_percentage": round(probability * 100, 1),
            "risk_category": risk_category,
            "risk_color": risk_color,
            "interpretation": interpretation,
            "contributing_factors": contributing_factors,
            "model_used": model_name,
            "input_summary": patient_data,
            "disclaimer": CLINICAL_DISCLAIMER,
        }

    def _analyze_risk_factors(self, data: Dict[str, Union[int, float]]) -> List[Dict[str, str]]:
        """Identify patient-specific clinical risk triggers based on evidence-based thresholds.

        Args:
            data: Patient input values.

        Returns:
            List of dictionaries with factor name, observed value, and clinical context.
        """
        factors: List[Dict[str, str]] = []

        # Fluoroscopy Vessels (ca)
        ca_val = float(data.get("ca", 0))
        if ca_val > 0:
            factors.append({
                "factor": "Major Coronary Vessels Calcified/Narrowed",
                "finding": f"{int(ca_val)} vessel(s) detected via fluoroscopy",
                "severity": "High",
                "detail": "Fluoroscopic opacification of coronary arteries indicates multi-vessel atherosclerosis.",
            })

        # Thallium Defect (thal)
        thal_val = float(data.get("thal", 3.0))
        if thal_val == 7.0 or thal_val == 7:
            factors.append({
                "factor": "Reversible Thallium-201 Perfusion Defect",
                "finding": "Reversible myocardial perfusion defect on nuclear scintigraphy",
                "severity": "High",
                "detail": "Indicates inducible myocardial ischemia during exercise stress.",
            })
        elif thal_val == 6.0 or thal_val == 6:
            factors.append({
                "factor": "Fixed Thallium-201 Defect",
                "finding": "Fixed myocardial defect on nuclear scintigraphy",
                "severity": "Moderate",
                "detail": "Suggests prior myocardial infarction or permanent scar tissue.",
            })

        # ST Depression (oldpeak)
        oldpeak_val = float(data.get("oldpeak", 0.0))
        if oldpeak_val >= 1.5:
            factors.append({
                "factor": "Significant Exercise ST Depression",
                "finding": f"{oldpeak_val} mm ST depression relative to resting baseline",
                "severity": "High",
                "detail": "Substantial subendocardial ischemia induced by cardiac exertion.",
            })
        elif oldpeak_val >= 1.0:
            factors.append({
                "factor": "Moderate Exercise ST Depression",
                "finding": f"{oldpeak_val} mm ST depression",
                "severity": "Moderate",
                "detail": "Mild-to-moderate ischemic response during exercise stress testing.",
            })

        # Exercise-Induced Angina (exang)
        if int(data.get("exang", 0)) == 1:
            factors.append({
                "factor": "Exercise-Induced Angina",
                "finding": "Positive for exertional angina pectoris",
                "severity": "High",
                "detail": "Chest discomfort precipitated by exertion strongly correlates with coronary insufficiency.",
            })

        # Chest Pain Type (cp)
        cp_val = int(data.get("cp", 1))
        if cp_val == 4:
            factors.append({
                "factor": "Asymptomatic / Silent Angina Presentation",
                "finding": "Type 4 (Asymptomatic)",
                "severity": "Moderate",
                "detail": "In the Cleveland cohort, asymptomatic patients undergoing catheterization frequently displayed severe silent multi-vessel disease.",
            })

        # Resting Blood Pressure (trestbps)
        trestbps_val = int(data.get("trestbps", 120))
        if trestbps_val >= 140:
            factors.append({
                "factor": "Stage 2 Hypertension",
                "finding": f"Resting systolic BP {trestbps_val} mm Hg",
                "severity": "Moderate",
                "detail": "Sustained hypertension exacerbates arterial shear stress and endothelial dysfunction.",
            })

        # Serum Cholesterol (chol)
        chol_val = int(data.get("chol", 200))
        if chol_val >= 240:
            factors.append({
                "factor": "Hypercholesterolemia",
                "finding": f"Serum cholesterol {chol_val} mg/dL (High risk >= 240 mg/dL)",
                "severity": "Moderate",
                "detail": "Elevated circulating lipids accelerate atherosclerotic plaque progression.",
            })

        # Age & Chronotropic Competence
        age_val = int(data.get("age", 50))
        thalach_val = int(data.get("thalach", 150))
        expected_max_hr = 220 - age_val
        if thalach_val < 0.70 * expected_max_hr:
            factors.append({
                "factor": "Reduced Chronotropic Response",
                "finding": f"Max HR {thalach_val} bpm (< 70% of age-predicted {expected_max_hr} bpm)",
                "severity": "Low",
                "detail": "Inability to achieve adequate peak heart rate during stress testing.",
            })

        return factors


if __name__ == "__main__":
    predictor = HeartDiseasePredictor()
    # Test sample patient: High risk
    sample_patient = {
        "age": 67,
        "sex": 1,
        "cp": 4,
        "trestbps": 160,
        "chol": 286,
        "fbs": 0,
        "restecg": 2,
        "thalach": 108,
        "exang": 1,
        "oldpeak": 1.5,
        "slope": 2,
        "ca": 3.0,
        "thal": 3.0,
    }
    result = predictor.predict_patient(sample_patient)
    print("\nInference Test Result:")
    print(f"Prediction:     {result['prediction_label']}")
    print(f"Risk Score:     {result['risk_percentage']}%")
    print(f"Risk Category:  {result['risk_category']}")
    print("Identified Contributing Factors:")
    for f in result["contributing_factors"]:
        print(f"  - [{f['severity']}] {f['factor']}: {f['finding']}")
