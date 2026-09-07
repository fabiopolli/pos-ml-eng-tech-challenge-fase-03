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

    result = airflow_pipeline.validate_dataset_file(dataset, sample_size=2_000, random_state=42)

    assert result["prepared_rows"] == 2_000
    assert result["classes"] == [1, 2, 3, 4, 5]
    assert set(result) == {
        "dataset_path",
        "dataset_sha256",
        "input_rows",
        "missing_or_empty_rows",
        "conflicting_texts",
        "conflicting_rows",
        "duplicate_rows",
        "eligible_rows",
        "prepared_rows",
        "sample_size",
        "random_state",
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


def test_ingestion_requires_complete_git_credentials(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provided together"):
        airflow_pipeline.ingest_from_git(
            repository_url="https://example.invalid/repository.git",
            branch="main",
            dataset_relative_path="data/dataset.csv",
            destination=tmp_path / "dataset.csv",
            git_username="user",
        )


def test_ingestion_keeps_git_token_out_of_command(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.csv"
    source.write_text("condition_label,medical_abstract\n", encoding="utf-8")
    captured_envs = []
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        # Capture a snapshot of the env as received by the subprocess, because
        # the production code wipes credentials from the in-memory dict after
        # the subprocess exits.
        captured_envs.append(dict(kwargs.get("env", {})))
        if command[1] == "clone":
            checkout = Path(command[-1])
            (checkout / "data").mkdir(parents=True)
            (checkout / "data" / "dataset.csv").write_bytes(source.read_bytes())
            return type("Result", (), {"stdout": ""})()
        return type("Result", (), {"stdout": "a" * 40 + "\n"})()

    monkeypatch.setattr(airflow_pipeline.subprocess, "run", fake_run)

    airflow_pipeline.ingest_from_git(
        repository_url="https://dagshub.com/example/project.git",
        branch="main",
        dataset_relative_path="data/dataset.csv",
        destination=tmp_path / "published.csv",
        git_username="example-user",
        git_token="secret-token",
    )

    clone_command = calls[0][0]
    clone_env = captured_envs[0]
    assert "secret-token" not in " ".join(clone_command)
    assert clone_env["DAGSHUB_USER_TOKEN"] == "secret-token"
    assert clone_env["GIT_TERMINAL_PROMPT"] == "0"
    assert clone_env["GIT_ASKPASS_REQUIRE"] == "force"
    assert Path(clone_env["GIT_ASKPASS"]).name == "git-askpass.sh"
    assert "PATH" in clone_env


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


def test_git_subprocess_errors_redact_credentials_in_stderr() -> None:
    """stderr/stdout from git must not surface ``user:token@`` segments."""

    import subprocess

    from triage_ml.orchestration import airflow_pipeline

    fake_stderr = (
        "fatal: Authentication failed for "
        "'https://alice:s3cr3t-token@dagshub.com/example/project.git/'\n"
    )

    def fake_run(*args, **kwargs):
        error = subprocess.CalledProcessError(128, args[0])
        error.stderr = fake_stderr
        error.stdout = ""
        raise error

    original = airflow_pipeline.subprocess.run
    airflow_pipeline.subprocess.run = fake_run  # type: ignore[attr-defined]
    try:
        with pytest.raises(RuntimeError) as excinfo:
            airflow_pipeline._run_git(
                ["git", "clone", "https://example.invalid/repo.git"],
                environment={"PATH": "/bin"},
                timeout=30,
            )
    finally:
        airflow_pipeline.subprocess.run = original  # type: ignore[attr-defined]
    message = str(excinfo.value)
    assert "s3cr3t-token" not in message
    assert "[REDACTED]" in message


def test_ingestion_refuses_destination_through_symlink(tmp_path: Path) -> None:
    """A symlink in any ancestor of the destination must abort the publish step."""

    link = tmp_path / "linkdir"
    link.symlink_to(tmp_path / "real")
    target = link / "dataset.csv"

    with pytest.raises(RuntimeError, match="symlink"):
        airflow_pipeline.ingest_from_git(
            repository_url="https://example.invalid/repository.git",
            branch="main",
            dataset_relative_path="data/dataset.csv",
            destination=target,
        )


def test_validate_dataset_file_aligns_with_training_config(tmp_path: Path) -> None:
    """validate_dataset_file must read sample_size/random_state from the YAML."""

    dataset = tmp_path / "dataset.csv"
    _raw_dataset(rows_per_label=400).to_csv(dataset, index=False)
    config = tmp_path / "training.yaml"
    config.write_text("random_state: 7\nsample_size: 2000\n", encoding="utf-8")

    result = airflow_pipeline.validate_dataset_file(dataset, config_path=config)

    assert result["sample_size"] == 2000
    assert result["random_state"] == 7
