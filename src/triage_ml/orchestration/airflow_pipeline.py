"""Safe, testable building blocks for the Airflow retraining DAG."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from triage_ml.data.prepare import prepare_dataset
from triage_ml.models.artifact import validate_artifact_bundle
from triage_ml.models.train import run_training


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it entirely into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("dataset_relative_path must be a safe repository-relative path")
    return relative


def ingest_from_git(
    *,
    repository_url: str,
    branch: str,
    dataset_relative_path: str,
    destination: str | Path,
    git_username: str | None = None,
    git_token: str | None = None,
) -> dict[str, Any]:
    """Clone a data source in isolation and atomically publish only its dataset."""

    if not repository_url.startswith("https://"):
        raise ValueError("repository_url must use HTTPS")
    if not branch or branch.startswith("-"):
        raise ValueError("branch must be a non-empty Git branch name")
    if bool(git_username) != bool(git_token):
        raise ValueError("git_username and git_token must be provided together")
    relative = _safe_relative_path(dataset_relative_path)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="triage-airflow-ingest-") as temp_dir:
        checkout = Path(temp_dir) / "source"
        clone_environment = os.environ.copy()
        clone_environment["GIT_TERMINAL_PROMPT"] = "0"
        if git_username and git_token:
            askpass = Path(temp_dir) / "git-askpass.sh"
            askpass.write_text(
                "#!/bin/sh\n"
                'case "$1" in\n'
                "  *Username*) printf '%s\\n' \"$DAGSHUB_USERNAME\" ;;\n"
                "  *Password*) printf '%s\\n' \"$DAGSHUB_USER_TOKEN\" ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            askpass.chmod(0o700)
            clone_environment.update(
                {
                    "GIT_ASKPASS": str(askpass),
                    "DAGSHUB_USERNAME": git_username,
                    "DAGSHUB_USER_TOKEN": git_token,
                }
            )
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--depth",
                "1",
                "--single-branch",
                "--branch",
                branch,
                repository_url,
                str(checkout),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
            env=clone_environment,
        )
        source = checkout.joinpath(*relative.parts)
        if not source.is_file():
            raise FileNotFoundError(f"dataset not found in repository: {relative}")
        commit = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        staged = destination.with_name(f".{destination.name}.tmp")
        try:
            shutil.copyfile(source, staged)
            os.replace(staged, destination)
        finally:
            staged.unlink(missing_ok=True)

    return {
        "dataset_path": str(destination),
        "dataset_sha256": file_sha256(destination),
        "source_commit": commit,
        "source_branch": branch,
    }


def validate_dataset_file(
    dataset_path: str | Path, *, sample_size: int = 5_000, random_state: int = 42
) -> dict[str, Any]:
    """Validate the canonical data contract and return only non-sensitive metadata."""

    dataset_path = Path(dataset_path)
    raw = pd.read_csv(dataset_path)
    prepared, report = prepare_dataset(raw, sample_size=sample_size, random_state=random_state)
    return {
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "input_rows": report.input_rows,
        "eligible_rows": report.eligible_rows,
        "prepared_rows": len(prepared),
        "classes": sorted(int(value) for value in prepared["target"].unique()),
    }


def find_reusable_artifact(
    models_dir: str | Path, *, dataset_sha256: str, config_file_sha256: str
) -> dict[str, Any] | None:
    """Find a prior successful orchestration run with identical declared inputs."""

    for manifest_path in sorted(Path(models_dir).glob("*/airflow_run.json"), reverse=True):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("dataset_sha256") == dataset_sha256
                and manifest.get("config_file_sha256") == config_file_sha256
            ):
                joblib_path = manifest_path.parent / "model.joblib"
                metadata = validate_artifact_bundle(joblib_path)
                return {
                    "reused": True,
                    "model_version": metadata["model_version"],
                    "joblib": str(joblib_path),
                    "metrics": metadata["metrics"],
                }
        except (OSError, ValueError, RuntimeError, KeyError):
            continue
    return None


def train_evaluate_persist(
    *,
    dataset_path: str | Path,
    models_dir: str | Path,
    figures_dir: str | Path,
    config_path: str | Path,
    source_commit: str,
) -> dict[str, Any]:
    """Reuse an identical valid run or execute the project's canonical trainer."""

    dataset_hash = file_sha256(dataset_path)
    config_hash = file_sha256(config_path)
    reusable = find_reusable_artifact(
        models_dir,
        dataset_sha256=dataset_hash,
        config_file_sha256=config_hash,
    )
    if reusable is not None:
        return reusable

    summary = run_training(
        raw_csv_path=dataset_path,
        out_dir=models_dir,
        figures_dir=figures_dir,
        config_path=config_path,
    )
    version_dir = Path(models_dir) / summary["model_version"]
    run_manifest = {
        "dataset_sha256": dataset_hash,
        "config_file_sha256": config_hash,
        "source_commit": source_commit,
    }
    (version_dir / "airflow_run.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "reused": False,
        "model_version": summary["model_version"],
        "joblib": summary["paths"]["joblib"],
        "metrics": summary["metrics"],
    }


def validate_training_output(joblib_path: str | Path) -> dict[str, Any]:
    """Validate the persisted bundle without deserializing its model payload."""

    metadata = validate_artifact_bundle(joblib_path)
    return {
        "model_version": metadata["model_version"],
        "metrics": metadata["metrics"],
        "checksum_sha256": metadata["checksum_sha256"],
    }
