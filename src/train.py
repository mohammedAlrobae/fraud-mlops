"""
src/train.py

Train an XGBoost fraud detection model tracked with MLflow.
Contains the reusable `train_model()` function used by both baseline training and Optuna tuning.
"""

import os
import time
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)

# ── Paths ────────────────────────────────────────────────────────────────────
TRAIN_PATH = "data/processed/train.csv"
VAL_PATH   = "data/processed/validation.csv"

# Columns that are NOT features
NON_FEATURE_COLS = ["isFraud", "transaction_id", "step"]

# ── MLflow Settings ───────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME     = "fraud-detection"

# ── Baseline Hyperparameters ──────────────────────────────────────────────────
PARAMS = {
    "objective": "binary:logistic",
    "n_estimators"  : 300,
    "max_depth"     : 6,
    "learning_rate" : 0.1,
    "eval_metric"   : "aucpr",
    "random_state"  : 42,
    "n_jobs"        : -1,
}


def load_data(path: str):
    """Return (X, y) for a given processed CSV."""
    df = pd.read_csv(path)
    X = df.drop(columns=NON_FEATURE_COLS)
    y = df["isFraud"]
    return X, y


def train_model(
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    tag: str = None,
    nested: bool = False,
    run_name: str = None,
):
    """
    Train an XGBoost model, evaluate on validation set, and log params, metrics,
    signature, and input example to MLflow.

    Returns:
        (model, metrics, run_id)
    """
    params = params.copy()

    # Compute scale_pos_weight dynamically if not explicitly provided
    if "scale_pos_weight" not in params:
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        params["scale_pos_weight"] = round(n_neg / n_pos, 2)

    with mlflow.start_run(nested=nested, run_name=run_name) as run:
        if tag:
            mlflow.set_tag("stage", tag)

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_pred       = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        metrics = {
            "precision" : round(precision_score(y_val, y_pred,       zero_division=0), 4),
            "recall"    : round(recall_score(y_val, y_pred,           zero_division=0), 4),
            "f1"        : round(f1_score(y_val, y_pred,               zero_division=0), 4),
            "pr_auc"    : round(average_precision_score(y_val, y_pred_proba),           4),
            "roc_auc"   : round(roc_auc_score(y_val, y_pred_proba),                     4),
        }

        # Log params & metrics
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Log model with signature and input example (Always)
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(3)
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        return model, metrics, run.info.run_id


def main():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("1. Loading train and validation sets...")
    X_train, y_train = load_data(TRAIN_PATH)
    X_val,   y_val   = load_data(VAL_PATH)
    print(f"   Train : {len(X_train):,} rows | {y_train.sum():,} fraud cases")
    print(f"   Val   : {len(X_val):,} rows | {y_val.sum():,} fraud cases")

    # ── 2. MLflow Setup ───────────────────────────────────────────────────────
    print("\n2. Setting up MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── 3. Train Baseline Model ───────────────────────────────────────────────
    print("\n3. Training baseline XGBoost model...")
    start_time = time.time()
    model, metrics, run_id = train_model(
        PARAMS, X_train, y_train, X_val, y_val, tag="baseline", run_name="baseline_model"
    )
    elapsed_sec = time.time() - start_time
    print(f"   Training took: {elapsed_sec:.2f} seconds")

    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # ── 4. Threshold Analysis ─────────────────────────────────────────────────
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_results = []
    best_f1 = -1.0
    best_thresh = 0.5

    for t in thresholds:
        t_pred = (y_pred_proba >= t).astype(int)
        p = precision_score(y_val, t_pred, zero_division=0)
        r = recall_score(y_val, t_pred, zero_division=0)
        f = f1_score(y_val, t_pred, zero_division=0)
        threshold_results.append({
            "threshold": t,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4)
        })
        if f > best_f1:
            best_f1 = f
            best_thresh = t

    # ── 5. Print Results ──────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("         VALIDATION METRICS (Default Threshold = 0.5)")
    print("=" * 50)
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC    : {metrics['pr_auc']:.4f}  ← PRIMARY METRIC")
    print("=" * 50)

    print("\n" + "=" * 54)
    print("                 THRESHOLD ANALYSIS")
    print("=" * 54)
    print(f"  {'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 54)
    for res in threshold_results:
        marker = "  ← Optimal F1" if res["threshold"] == best_thresh else ""
        print(f"  {res['threshold']:<10.1f} | {res['precision']:<10.4f} | {res['recall']:<10.4f} | {res['f1']:<10.4f}{marker}")
    print("=" * 54)
    print(f"  Optimal F1 Threshold : {best_thresh:.1f} (F1 = {best_f1:.4f})")
    print(f"\n  ✅ Final PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"  📁 MLflow run ID: {run_id}")


if __name__ == "__main__":
    main()
