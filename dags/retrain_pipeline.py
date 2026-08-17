"""
dags/retrain_pipeline.py

This DAG represents the scheduled retraining step of the MLOps lifecycle (Track 3 from our original design).
In production environments, this pipeline will be triggered automatically by model drift detection alerts
or scheduled intervals. Currently set up for manual trigger.
"""

from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "start_date": datetime(2026, 1, 1),
    "catchup": False,
}

with DAG(
    dag_id="fraud_retrain_pipeline",
    default_args=default_args,
    schedule=None,
    catchup=False,
    doc_md=__doc__,
) as dag:

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command="cd /opt/airflow/project && python3 src/preprocess.py",
    )

    train_baseline = BashOperator(
        task_id="train_baseline",
        bash_command="cd /opt/airflow/project && python3 src/train.py",
    )

    preprocess >> train_baseline
