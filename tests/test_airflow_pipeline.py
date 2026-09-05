import json
from pathlib import Path

import pandas as pd
import pytest

from triage_ml.orchestration import airflow_pipeline


def _raw_dataset(rows_per_label: int = 400) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "medical_abstract": [
                f"Unique sufficiently descriptive medical abstract {label} row {row}"
                for label in range(1, 6)
                for row in range(rows_per_label)
            ],
            "condition_label": [label for label in range(1, 6) for _ in range(rows_per_label)],
        }
    )


def test_validate_dataset_file_returns_only_metadata(tmp_path: Path) -> None:
    dataset = tmp_path / "medical.csv"
    _raw_dataset().to_csv(dataset, index=False)

    result = airflow_pipeline.validate_dataset_file(dataset, sample_size=2_000)

    assert result["prepared_rows"] == 2_000
    assert result["classes"] == [1, 2, 3, 4, 5]
    assert set(result) == {
        "dataset_path",
        "dataset_sha256",
        "input_rows",
        "eligible_rows",
        "prepared_rows",
        "classes",
    }
    assert "medical_abstract" not in json.dumps(result)


@pytest.mark.parametrize("relative", ["../secret.csv", "/data/file.csv", ""])
def test_ingestion_rejects_unsafe_dataset_paths(relative: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repository-relative"):
        airflow_pipeline.ingest_from_git(
            repository_url="https://example.invalid/repository.git",
            branch="main",
            dataset_relative_path=relative,
            destination=tmp_path / "dataset.csv",
        )


def test_training_reuses_valid_identical_inputs(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "dataset.csv"
    config = tmp_path / "training.yaml"
    dataset.write_text("condition_label,medical_abstract\n", encoding="utf-8")
    config.write_text("random_state: 42\n", encoding="utf-8")
    expected = {
        "reused": True,
        "model_version": "20260905T000000Z-0123456789ab",
        "joblib": "model.joblib",
        "metrics": {"macro_f1": 0.75},
    }
    monkeypatch.setattr(airflow_pipeline, "find_reusable_artifact", lambda *a, **k: expected)
    monkeypatch.setattr(
        airflow_pipeline,
        "run_training",
        lambda **kwargs: pytest.fail("training must not run for identical inputs"),
    )

    result = airflow_pipeline.train_evaluate_persist(
        dataset_path=dataset,
        models_dir=tmp_path / "models",
        figures_dir=tmp_path / "figures",
        config_path=config,
        source_commit="a" * 40,
    )

    assert result == expected


def test_airflow_dag_source_compiles_and_declares_expected_policy() -> None:
    dag_path = Path(__file__).parents[1] / "airflow" / "dags" / "triage_retraining.py"
    source = dag_path.read_text(encoding="utf-8")

    compile(source, str(dag_path), "exec")
    assert 'dag_id="triage_ml_retraining"' in source
    assert "schedule=None" in source
    assert "catchup=False" in source
    assert "max_active_runs=1" in source
