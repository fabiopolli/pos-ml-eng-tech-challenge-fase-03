"""Tests for ``train_with_sample_size`` — optimization DAG helper."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from triage_ml.orchestration import airflow_pipeline


def _make_csv(path: Path, *, rows: int) -> Path:
    rows_per_label = rows // 4
    label_topics = {
        1: "cardiology neoplasms tumor",
        2: "cardiology heart valve",
        3: "nervous brain neuron",
        4: "general pathology diagnosis",
    }
    rows_data = []
    for label in range(1, 5):
        for index in range(rows_per_label):
            rows_data.append(
                {
                    "medical_abstract": (
                        f"medical abstract {label} number {index:05d} about "
                        f"{label_topics[label]} case study {index}"
                    ),
                    "condition_label": label,
                }
            )
    df = pd.DataFrame(rows_data)
    df.to_csv(path, index=False)
    return path


def _write_config(path: Path, *, sample_size: int) -> None:
    config = {
        "sample_size": sample_size,
        "random_state": 7,
        "test_size": 0.2,
        "cv_folds": 2,
        "language": "en",
        "task_type": "multiclass_text_classification",
        "label_mapping": {"1": "neoplasms", "2": "cardio", "3": "nervous", "4": "general"},
        "tfidf": {"ngram_range": [1, 1], "min_df": 2},
        "logreg": {"C": 1.0, "solver": "lbfgs", "max_iter": 200, "class_weight": "balanced"},
        "linear_svc": {"C": 1.0, "class_weight": "balanced"},
        "model_name": "triage_test",
    }
    path.write_text(json.dumps(config), encoding="utf-8")


def test_train_with_sample_size_persists_manifest(tmp_path: Path) -> None:
    raw_csv = _make_csv(tmp_path / "train.csv", rows=8_000)
    config_path = tmp_path / "training.yaml"
    _write_config(config_path, sample_size=4_000)
    models_dir = tmp_path / "models"
    figures_dir = tmp_path / "figures"

    summary = airflow_pipeline.train_with_sample_size(
        dataset_path=raw_csv,
        sample_size=4_000,
        models_dir=models_dir,
        figures_dir=figures_dir,
        config_path=config_path,
        source_commit="0" * 40,
    )

    assert summary["reused"] is False
    assert summary["sample_size"] == 4_000
    assert Path(summary["joblib"]).is_file()
    version_dir = Path(summary["joblib"]).parent
    assert (version_dir / "airflow_run.json").is_file()
    manifest = json.loads((version_dir / "airflow_run.json").read_text(encoding="utf-8"))
    assert manifest["sample_size"] == 4_000
    assert manifest["dataset_sha256"]
    assert manifest["config_file_sha256"]


def test_train_with_sample_size_reuses_existing_artifact(tmp_path: Path) -> None:
    raw_csv = _make_csv(tmp_path / "train.csv", rows=8_000)
    config_path = tmp_path / "training.yaml"
    _write_config(config_path, sample_size=4_000)

    first = airflow_pipeline.train_with_sample_size(
        dataset_path=raw_csv,
        sample_size=4_000,
        models_dir=tmp_path / "models",
        figures_dir=tmp_path / "figures",
        config_path=config_path,
        source_commit="0" * 40,
    )

    second = airflow_pipeline.train_with_sample_size(
        dataset_path=raw_csv,
        sample_size=4_000,
        models_dir=tmp_path / "models",
        figures_dir=tmp_path / "figures",
        config_path=config_path,
        source_commit="0" * 40,
    )

    assert second["reused"] is True
    assert second["model_version"] == first["model_version"]


def test_train_with_sample_size_rejects_invalid_size(tmp_path: Path) -> None:
    raw_csv = _make_csv(tmp_path / "train.csv", rows=8_000)
    config_path = tmp_path / "training.yaml"
    _write_config(config_path, sample_size=4_000)

    with pytest.raises(ValueError):
        airflow_pipeline.train_with_sample_size(
            dataset_path=raw_csv,
            sample_size=999,
            models_dir=tmp_path / "models",
            figures_dir=tmp_path / "figures",
            config_path=config_path,
            source_commit="0" * 40,
        )
