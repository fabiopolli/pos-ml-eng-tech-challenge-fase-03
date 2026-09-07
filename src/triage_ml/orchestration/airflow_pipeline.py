"""Safe, testable building blocks for the Airflow retraining DAG."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from triage_ml.data.prepare import prepare_dataset
from triage_ml.models.artifact import VERSION_PATTERN, validate_artifact_bundle
from triage_ml.models.train import load_config, run_training

MAX_DATASET_BYTES = 200 * 1024 * 1024
MAX_MANIFEST_BYTES = 1 * 1024 * 1024
# Match either inline userinfo (``scheme://user:token@``) or env-var leaks
# in git stderr (``DAGSHUB_USERNAME=...`` / ``DAGSHUB_USER_TOKEN=...``).
_CREDENTIAL_PATTERN = re.compile(
    r"(://)([^/\s:@]+):([^@\s/]+)@|(?P<env>DAGSHUB_(?:USERNAME|USER_TOKEN)=[^\s/]+)"
)


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


def _redact_credentials(text: str) -> str:
    """Strip accidental credentials from command stderr/stdout.

    Covers two leak channels:

    * ``scheme://user:token@host`` URLs that git may echo back on auth
      failure (replaced with ``scheme://[REDACTED]@host``).
    * ``DAGSHUB_USERNAME=...`` and ``DAGSHUB_USER_TOKEN=...`` env-var
      fragments that can appear in hooks or proxy error messages.
    """

    return _CREDENTIAL_PATTERN.sub(
        lambda match: "[REDACTED]" if match.group("env") else f"{match.group(1)}[REDACTED]@",
        text,
    )


def _ensure_no_symlink_ancestor(path: Path) -> None:
    """Refuse to write through any symlink in the path's ancestry."""

    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise RuntimeError(f"refusing to operate through symlink: {ancestor}")


def _git_environment(
    askpass: Path | None, *, username: str | None, token: str | None
) -> dict[str, str]:
    """Build the minimal environment for the git subprocess (no inherited secrets)."""

    environment: dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "LC_ALL": "C.UTF-8",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS_REQUIRE": "force",
    }
    if askpass is not None and username and token:
        environment["GIT_ASKPASS"] = str(askpass)
        environment["DAGSHUB_USERNAME"] = username
        environment["DAGSHUB_USER_TOKEN"] = token
    return environment


def _run_git(
    command: list[str],
    *,
    environment: dict[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except subprocess.CalledProcessError as exc:
        stderr = _redact_credentials(exc.stderr or "")
        stdout = _redact_credentials(exc.stdout or "")
        raise RuntimeError(
            f"git command failed (exit {exc.returncode}): {' '.join(command[:4])}... "
            f"stderr={stderr.strip()} stdout={stdout.strip()}"
        ) from exc


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
    _ensure_no_symlink_ancestor(destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="triage-airflow-ingest-") as temp_dir:
        checkout = Path(temp_dir) / "source"
        askpass: Path | None = None
        if git_username and git_token:
            askpass = Path(temp_dir) / "git-askpass.sh"
            askpass.write_text(
                "#!/bin/sh\n"
                "set -eu\n"
                'case "$1" in\n'
                '  *Username*) printf "%s\\n" "$DAGSHUB_USERNAME" ;;\n'
                '  *Password*) printf "%s\\n" "$DAGSHUB_USER_TOKEN" ;;\n'
                "  *) exit 1 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            askpass.chmod(0o700)
        clone_environment = _git_environment(askpass, username=git_username, token=git_token)
        try:
            _run_git(
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
                environment=clone_environment,
                timeout=300,
            )
            source = checkout.joinpath(*relative.parts)
            if not source.is_file():
                raise FileNotFoundError(f"dataset not found in repository: {relative}")
            commit = _run_git(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                environment=clone_environment,
                timeout=30,
            ).stdout.strip()
            staged = destination.with_name(f".{destination.name}.tmp")
            try:
                shutil.copyfile(source, staged)
                os.replace(staged, destination)
            finally:
                staged.unlink(missing_ok=True)
        finally:
            # Wipe the in-memory copy so the token does not linger after the
            # subprocess exits. The subprocess already received its own copy
            # of the value when ``env=`` was evaluated.
            clone_environment["DAGSHUB_USER_TOKEN"] = ""
            clone_environment["DAGSHUB_USERNAME"] = ""

    return {
        "dataset_path": str(destination),
        "dataset_sha256": file_sha256(destination),
        "source_commit": commit,
        "source_branch": branch,
    }


def _load_preparation_settings(
    config_path: str | Path, *, sample_size: int | None, random_state: int | None
) -> tuple[int, int]:
    """Resolve (sample_size, random_state) preferring explicit overrides over the YAML."""

    if sample_size is None or random_state is None:
        config = load_config(Path(config_path))
        if sample_size is None:
            sample_size = int(config.get("sample_size", 5000))
        if random_state is None:
            random_state = int(config.get("random_state", 42))
    return sample_size, random_state


def validate_dataset_file(
    dataset_path: str | Path,
    *,
    config_path: str | Path | None = None,
    sample_size: int | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Validate the canonical data contract and return only non-sensitive metadata."""

    dataset_path = Path(dataset_path)
    size = dataset_path.stat().st_size
    if size > MAX_DATASET_BYTES:
        raise ValueError(
            f"dataset exceeds {MAX_DATASET_BYTES} bytes limit; refusing to load into memory"
        )
    if config_path is not None:
        sample_size, random_state = _load_preparation_settings(
            config_path, sample_size=sample_size, random_state=random_state
        )
    else:
        sample_size = sample_size or 5_000
        random_state = random_state if random_state is not None else 42
    raw = pd.read_csv(dataset_path)
    prepared, report = prepare_dataset(raw, sample_size=sample_size, random_state=random_state)
    return {
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "input_rows": report.input_rows,
        "missing_or_empty_rows": report.missing_or_empty_rows,
        "conflicting_texts": report.conflicting_texts,
        "conflicting_rows": report.conflicting_rows,
        "duplicate_rows": report.duplicate_rows,
        "eligible_rows": report.eligible_rows,
        "prepared_rows": len(prepared),
        "sample_size": sample_size,
        "random_state": random_state,
        "classes": sorted(int(value) for value in prepared["target"].unique()),
    }


def find_reusable_artifact(
    models_dir: str | Path, *, dataset_sha256: str, config_file_sha256: str
) -> dict[str, Any] | None:
    """Find a prior successful orchestration run with identical declared inputs."""

    for manifest_path in sorted(Path(models_dir).glob("*/airflow_run.json"), reverse=True):
        if not VERSION_PATTERN.fullmatch(manifest_path.parent.name):
            continue
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            continue
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
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError):
            continue
    return None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_no_symlink_ancestor(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


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
    _atomic_write_json(version_dir / "airflow_run.json", run_manifest)
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
