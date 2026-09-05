"""Airflow DAG for isolated ingestion and reproducible model retraining."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

from airflow.sdk import dag, task

from triage_ml.orchestration.airflow_pipeline import (
    ingest_from_git,
    train_evaluate_persist,
    validate_dataset_file,
    validate_training_output,
)


@dag(
    dag_id="triage_ml_retraining",
    description="Ingest, validate, train, evaluate and persist the triage model",
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=2)},
    tags=["ml", "retraining", "triage"],
)
def triage_ml_retraining():
    @task(execution_timeout=timedelta(minutes=10))
    def ingest() -> dict:
        return ingest_from_git(
            repository_url=os.environ["DATA_REPOSITORY_URL"],
            branch=os.getenv("DATA_REPOSITORY_BRANCH", "main"),
            dataset_relative_path=os.getenv("DATASET_RELATIVE_PATH", "data/medical_tc_train.csv"),
            destination=os.getenv("TRIAGE_RAW_CSV", "/opt/triage-ml/data/medical_tc_train.csv"),
            git_username=os.getenv("DAGSHUB_USERNAME") or None,
            git_token=os.getenv("DAGSHUB_USER_TOKEN") or None,
        )

    @task(execution_timeout=timedelta(minutes=10))
    def validate(ingestion: dict) -> dict:
        result = validate_dataset_file(ingestion["dataset_path"])
        if result["dataset_sha256"] != ingestion["dataset_sha256"]:
            raise ValueError("dataset changed between ingestion and validation")
        return {**ingestion, **result}

    @task(execution_timeout=timedelta(hours=1))
    def train(validated: dict) -> dict:
        return train_evaluate_persist(
            dataset_path=validated["dataset_path"],
            models_dir=os.getenv("TRIAGE_MODELS_DIR", "/opt/triage-ml/models"),
            figures_dir=os.getenv("TRIAGE_REPORTS_DIR", "/opt/triage-ml/reports/figures"),
            config_path=os.getenv(
                "TRIAGE_TRAINING_CONFIG",
                "/opt/triage-ml/configs/training.yaml",
            ),
            source_commit=validated["source_commit"],
        )

    @task(execution_timeout=timedelta(minutes=5))
    def verify(training: dict) -> dict:
        return validate_training_output(training["joblib"])

    verify(train(validate(ingest())))


triage_ml_retraining()
