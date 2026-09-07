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


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"required environment variable is missing or empty: {name}")
    return value


def _require_auth_credentials() -> tuple[str, str]:
    if os.environ.get("TRIAGE_REQUIRE_AUTH", "false").lower() != "true":
        return "", ""
    username = os.environ.get("DAGSHUB_USERNAME")
    token = os.environ.get("DAGSHUB_USER_TOKEN")
    if not username or not token:
        raise ValueError(
            "TRIAGE_REQUIRE_AUTH=true requires DAGSHUB_USERNAME and DAGSHUB_USER_TOKEN"
        )
    return username, token


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
    repository_url = _require_env("DATA_REPOSITORY_URL")
    branch = os.environ.get("DATA_REPOSITORY_BRANCH", "main")
    dataset_relative_path = os.environ.get("DATASET_RELATIVE_PATH", "data/medical_tc_train.csv")
    raw_csv_path = os.environ.get("TRIAGE_RAW_CSV", "/opt/triage-ml/data/medical_tc_train.csv")
    config_path = os.environ.get("TRIAGE_TRAINING_CONFIG", "/opt/triage-ml/configs/training.yaml")
    models_dir = os.environ.get("TRIAGE_MODELS_DIR", "/opt/triage-ml/models")
    figures_dir = os.environ.get("TRIAGE_REPORTS_DIR", "/opt/triage-ml/reports/figures")
    require_auth_user, require_auth_token = _require_auth_credentials()

    @task(execution_timeout=timedelta(minutes=10))
    def ingest() -> dict:
        return ingest_from_git(
            repository_url=repository_url,
            branch=branch,
            dataset_relative_path=dataset_relative_path,
            destination=raw_csv_path,
            git_username=require_auth_user or None,
            git_token=require_auth_token or None,
        )

    @task(execution_timeout=timedelta(minutes=10))
    def validate(ingestion: dict) -> dict:
        result = validate_dataset_file(ingestion["dataset_path"], config_path=config_path)
        if result["dataset_sha256"] != ingestion["dataset_sha256"]:
            raise ValueError("dataset changed between ingestion and validation")
        return {**ingestion, **result}

    @task(execution_timeout=timedelta(hours=1))
    def train(validated: dict) -> dict:
        return train_evaluate_persist(
            dataset_path=validated["dataset_path"],
            models_dir=models_dir,
            figures_dir=figures_dir,
            config_path=config_path,
            source_commit=validated["source_commit"],
        )

    @task(execution_timeout=timedelta(minutes=5))
    def verify(training: dict) -> dict:
        return validate_training_output(training["joblib"])

    verify(train(validate(ingest())))


triage_ml_retraining()
