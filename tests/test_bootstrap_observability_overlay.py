"""Smoke tests for the observability overlay bootstrap helper.

These tests verify that ``scripts/bootstrap_observability_overlay.py``
correctly writes the 4 required variables for
``infra/docker-compose.yml`` and that ``docker compose config``
interpolates them without the ``required variable X is missing a
value`` error reported in the Etapa 6 review cycle.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "bootstrap_observability_overlay.py"
COMPOSE = REPO_ROOT / "infra" / "docker-compose.yml"
REQUIRED_KEYS = (
    "MODEL_VERSION",
    "TRIAGE_ML_API_KEY_SERVICE",
    "TRIAGE_ML_API_KEY_DOCTOR",
    "TRIAGE_ML_API_KEY_PATIENT",
    "GRAFANA_ADMIN_PASSWORD",
)


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in REQUIRED_KEYS:
        env.pop(key, None)
    return env


def _run_helper(*, model_version: str | None = None) -> subprocess.CompletedProcess[str]:
    args = [sys.executable, str(SCRIPT), "--force"]
    if model_version is not None:
        args.extend(["--model-version", model_version])
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
        env=_clean_env(),
        cwd=REPO_ROOT,
    )


@pytest.fixture
def env_file_cleanup():
    """Cleanup hook to remove the helper-written ``.env`` after a test."""
    path = REPO_ROOT / ".env"
    yield path
    path.unlink(missing_ok=True)


def test_helper_writes_required_keys(env_file_cleanup: Path) -> None:
    result = _run_helper(model_version="20260101T000000Z-0123456789ab")
    assert result.returncode == 0, result.stderr
    contents = env_file_cleanup.read_text(encoding="utf-8")
    for key in REQUIRED_KEYS:
        assert f"{key}=" in contents, f"missing {key} in {env_file_cleanup}"


def test_docker_compose_config_interpolates(env_file_cleanup: Path) -> None:
    if not shutil.which("docker"):
        pytest.skip("docker not available")
    result = _run_helper(model_version="20260101T000000Z-0123456789ab")
    assert result.returncode == 0, result.stderr
    assert env_file_cleanup.is_file()

    compose_result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE),
            "--env-file",
            str(env_file_cleanup),
            "config",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert compose_result.returncode == 0, (
        f"docker compose config failed: {compose_result.stderr}"
        f"\nstdout: {compose_result.stdout[:2000]}"
    )
    assert "required variable" not in compose_result.stderr, compose_result.stderr


def test_helper_detects_latest_model_version(env_file_cleanup: Path) -> None:
    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    sentinel = models_dir / "20990101T000000Z-aaaaaaaaaaaa"
    sentinel.mkdir(exist_ok=True)
    try:
        result = _run_helper()
        assert result.returncode == 0, result.stderr
        contents = env_file_cleanup.read_text(encoding="utf-8")
        assert "MODEL_VERSION=20990101T000000Z-aaaaaaaaaaaa" in contents, contents
    finally:
        sentinel.rmdir()
