"""
src/tune.py

Hyperparameter tuning for XGBoost fraud detection using Optuna and MLflow.
Uses `train_model` from `src.train` for trials and final model retraining.
Registers the winning model in MLflow Model Registry and sets the 'champion' alias.
"""

import os
import time
import pandas as pd
import optuna
import mlflow
from mlflow.tracking import MlflowClient
try:
    from train import (
        load_data,
        train_model,
        TRAIN_PATH,
        VAL_PATH,
        MLFLOW_TRACKING_URI,
        EXPERIMENT_NAME,
    )
except ModuleNotFoundError:
    from src.train import (
        load_data,
        train_model,
        TRAIN_PATH,
        VAL_PATH,
        MLFLOW_TRACKING_URI,
        EXPERIMENT_NAME,
    )

# Suppress Optuna default verbosity for clean custom progress updates
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_NAME = "fraud-detection-model"
ALIAS      = "champion"


def main():
    print("1. Loading train and validation sets...")
    X_train, y_train = load_data(TRAIN_PATH)
    X_val,   y_val   = load_data(VAL_PATH)
    print(f"   Train : {len(X_train):,} rows | {y_train.sum():,} fraud cases")
    print(f"   Val   : {len(X_val):,} rows | {y_val.sum():,} fraud cases")

    print("\n2. Setting up MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    start_time = time.time()

    def print_progress(study, trial):
        elapsed = time.time() - start_time
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        print("\n" + "=" * 60)
        print(f"  [TRIAL {trial.number + 1}/30 FINISHED]")
        print(f"  Trial PR-AUC : {trial.value:.4f}")
        print(f"  Best PR-AUC  : {study.best_value:.4f}")
        print(f"  Elapsed Time : {elapsed_min}m {elapsed_sec}s")
        print("=" * 60 + "\n")

    # Start parent MLflow run for the Optuna study
    with mlflow.start_run(run_name="optuna_study_parent") as parent_run:
        print(f"   MLflow Parent Run ID: {parent_run.info.run_id}")
        print("\n3. Starting Optuna hyperparameter study (30 trials)...")

        def objective(trial):
            params = {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "random_state": 42,
                "n_jobs": -1,
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }

            print(f"\n--- Starting Trial {trial.number + 1}/30 ---")
            print(f"Hyperparameters: max_depth={params['max_depth']}, lr={params['learning_rate']:.4f}, n_est={params['n_estimators']}, min_child={params['min_child_weight']}, subsample={params['subsample']:.2f}, colsample={params['colsample_bytree']:.2f}")

            _, metrics, _ = train_model(
                params, X_train, y_train, X_val, y_val,
                nested=True, run_name=f"trial_{trial.number + 1}"
            )
            return metrics["pr_auc"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30, callbacks=[print_progress])

    print("\n" + "=" * 60)
    print("              OPTUNA STUDY COMPLETED")
    print("=" * 60)
    print(f"Best Trial Number : {study.best_trial.number + 1}")
    print(f"Best Validation PR-AUC : {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for param_name, param_val in study.best_params.items():
        print(f"  {param_name:<20}: {param_val}")

    # ── 4. Retrain Final Model with Best Parameters ───────────────────────────
    print("\n4. Retraining final best model...")
    best_params = study.best_params.copy()
    best_params.update({
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
    })

    final_model, metrics, best_run_id = train_model(
        best_params, X_train, y_train, X_val, y_val,
        nested=False, tag="tuned_best", run_name="optuna_best_model"
    )

    # ── 5. Register Model Version & Update Champion Alias ─────────────────────
    print("\n5. Registering model in MLflow Model Registry...")
    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    print(f"   Setting alias '{ALIAS}' -> Model Version v{mv.version}...")
    client.set_registered_model_alias(name=MODEL_NAME, alias=ALIAS, version=mv.version)

    description = (
        f"Winning Optuna-tuned XGBoost model (PR-AUC {metrics['pr_auc']:.4f}) logged with "
        "complete MLflow signature & input example. Assigned 'champion' alias."
    )
    client.update_model_version(
        name=MODEL_NAME,
        version=mv.version,
        description=description
    )

    print("\n" + "=" * 60)
    print("         CHAMPION MODEL REGISTRATION SUCCESSFUL")
    print("=" * 60)
    print(f"  Registered Model Name : {mv.name}")
    print(f"  Model Version         : v{mv.version}")
    print(f"  Alias Set             : '{ALIAS}' -> Version v{mv.version}")
    print(f"  Run ID                : {best_run_id}")
    print("=" * 60)
    print(f"\n📌 Model Loading URI (referenced by alias):")
    print(f"   models:/{MODEL_NAME}@{ALIAS}\n")


if __name__ == "__main__":
    main()
