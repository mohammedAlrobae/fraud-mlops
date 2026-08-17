"""
src/service.py

BentoML service for real-time fraud detection using the MLflow Champion model.
Applies shared preprocessing feature engineering and thresholding (0.9).
"""

import mlflow
import mlflow.xgboost
import pandas as pd
import bentoml
from pydantic import BaseModel

try:
    from src.preprocess import build_features
except ModuleNotFoundError:
    from preprocess import build_features

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_URI = "models:/fraud-detection-model@champion"


class TransactionRequest(BaseModel):
    step: int
    type: str  # e.g., "PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"
    amount: float
    oldbalanceOrg: float
    nameDest: str
    oldbalanceDest: float


@bentoml.service(
    name="fraud_detection_service",
)
class FraudDetectionService:
    def __init__(self):
        print(f"Connecting to MLflow: {MLFLOW_TRACKING_URI}...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        print(f"Loading champion model from: {MODEL_URI}...")
        self.model = mlflow.xgboost.load_model(MODEL_URI)

        print("Fetching model signature input columns from MLflow...")
        model_info = mlflow.models.get_model_info(MODEL_URI)
        if model_info.signature and model_info.signature.inputs:
            inputs = model_info.signature.inputs
            self.expected_columns = inputs.input_names() if callable(inputs.input_names) else inputs.input_names
        else:
            self.expected_columns = self.model.get_booster().feature_names

        print(f"Model signature expected columns ({len(self.expected_columns)}): {self.expected_columns}")

    @bentoml.api
    def predict(self, request: TransactionRequest) -> dict:
        """
        Predict fraud probability for an incoming transaction request.
        """
        # Convert request to single-row pandas DataFrame
        raw_df = pd.DataFrame([request.model_dump()])

        # Build features using shared feature engineering function
        features_df = build_features(raw_df)

        # Align columns to match model's expected signature columns exactly
        aligned_df = features_df.reindex(columns=self.expected_columns, fill_value=0)

        # Predict fraud probability using native XGBoost model
        fraud_proba = float(self.model.predict_proba(aligned_df)[:, 1][0])
        threshold = 0.9
        is_fraud = bool(fraud_proba >= threshold)

        return {
            "fraud_probability": round(fraud_proba, 6),
            "is_fraud": is_fraud,
            "threshold_used": threshold,
        }
