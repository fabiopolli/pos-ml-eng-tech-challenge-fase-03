"""Airflow DAG for Fase 2 / Etapa 5 — dataset sizing + ONNX optimization.

Mirrors the data-source invariants of ``triage_retraining`` (no inherited
secrets, ``GIT_ASKPASS`` for credentials, atomic publish) but iterates over
the ``dataset_sizing`` catalog from ``configs/training.yaml`` (or
``TRIAGE_DATASET_SLICES`` override) and finishes each slice with an ONNX
export + a benchmark comparing sklearn and ONNX variants.

The DAG is gated by ``TRIAGE_OPTIMIZATION_ENABLED`` (default ``false``) so
the ``docker-compose.airflow.yml`` from Etapa 7 keeps working without code
changes. ``optimization_enabled()`` re-reads the env var on every call so
operators can flip it at runtime via the Airflow Variables API.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml
from airflow.sdk import dag, task

from triage_ml.optimization.benchmark import (
    benchmark_predictor,
)
from triage_ml.optimization.dataloader import iter_dataset_slices
from triage_ml.optimization.registry import resolve_variant_loader
from triage_ml.orchestration.airflow_pipeline import (
    _atomic_write_json,
    benchmark_for_version,
    export_onnx_for_version,
    ingest_from_git,
    train_with_sample_size,
    validate_dataset_file,
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


def _dataset_sizing_from_config(config_path: str) -> list[int]:
    """Return the slice catalog from the training YAML, deferred to ``iter_dataset_slices``."""

    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return [slice_entry.sample_size for slice_entry in iter_dataset_slices(config)]


def _optimization_enabled() -> bool:
    """Read the gate on every invocation to honour runtime overrides."""

    return os.environ.get("TRIAGE_OPTIMIZATION_ENABLED", "false").lower() == "true"


def _sklearn_only_benchmark(version_dir: Path) -> dict:
    """Emit a sklearn-only benchmark JSON when the optimization path is off.

    The aggregate path uses ``_atomic_write_json`` to avoid last-write-wins
    races when multiple slices run in parallel.
    """

    probe = [
        f"Optimisation probe input number {i:04d} about cardiology and oncology" for i in range(8)
    ]
    predictor = resolve_variant_loader("sklearn")(version_dir / "model.joblib")
    result = benchmark_predictor(predictor, texts=probe, variant="sklearn", repetitions=5, warmup=1)
    return {
        "model_version": version_dir.name,
        "sklearn": result.to_dict(),
        "onnx": None,
    }


@dag(
    dag_id="triage_ml_retraining_optimization",
    description=(
        "Iterate dataset_sizing slices, train, persist the artifact, export "
        "ONNX and benchmark sklearn vs ONNX variants for each slice."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=2)},
    tags=["ml", "optimization", "triage"],
)
def triage_ml_retraining_optimization():
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

    @task(execution_timeout=timedelta(minutes=15))
    def validate(ingestion: dict) -> dict:
        result = validate_dataset_file(ingestion["dataset_path"], config_path=config_path)
        if result["dataset_sha256"] != ingestion["dataset_sha256"]:
            raise ValueError("dataset changed between ingestion and validation")
        return {**ingestion, **result}

    @task(execution_timeout=timedelta(hours=1))
    def train_slice(validated: dict, sample_size: int) -> dict:
        return train_with_sample_size(
            dataset_path=validated["dataset_path"],
            sample_size=sample_size,
            models_dir=models_dir,
            figures_dir=figures_dir,
            config_path=config_path,
            source_commit=validated["source_commit"],
        )

    @task(execution_timeout=timedelta(minutes=10))
    def export_onnx_for_slice(training: dict) -> dict:
        from triage_ml.optimization.optimize import ONNX_AVAILABLE

        result = {**training, "sample_size": training["sample_size"]}
        if not _optimization_enabled():
            return {**result, "optimization_skipped": True, "reason": "disabled"}
        if not ONNX_AVAILABLE:
            return {
                **result,
                "optimization_skipped": True,
                "reason": "missing optimization extras",
            }
        try:
            payload = export_onnx_for_version(Path(training["joblib"]).parent, opset=17)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            # The optimization extras are installed but the export itself
            # failed (broken model.onnx, operator-skipped slice, etc.).
            # Bubble the exception to surface it in the Airflow log; the
            # optimisation should never be silently dropped here because
            # the skipping already happened in the ``ONNX_AVAILABLE``
            # branch above.
            raise RuntimeError(
                "export_onnx_for_version failed for slice "
                f"{training.get('sample_size', '?')}: {exc!s}"
            ) from exc
        return {**result, **payload}

    @task(execution_timeout=timedelta(minutes=10))
    def benchmark_for_slice(optimized: dict) -> dict:
        version_dir = Path(optimized["joblib"]).parent
        if not _optimization_enabled() or optimized.get("optimization_skipped"):
            return _sklearn_only_benchmark(version_dir)
        try:
            payload = benchmark_for_version(version_dir)
        except RuntimeError:
            return _sklearn_only_benchmark(version_dir)
        return payload

    @task(execution_timeout=timedelta(minutes=5))
    def verify(slice_result: dict, sample_size: int) -> dict:
        from triage_ml.models.artifact import validate_artifact_bundle

        validate_artifact_bundle(slice_result["joblib"])
        # Persist a per-slice payload so the comparison task can read it
        # back without relying on XCom (which would blow the 48 KB
        # default cap with many slices).
        payload_path = (
            Path(os.environ.get("TRIAGE_REPORTS_DIR", "/opt/triage-ml/reports"))
            / "slices"
            / f"{sample_size}.json"
        )
        _atomic_write_json(payload_path, {"sample_size": sample_size, **slice_result})
        return {"sample_size": sample_size, "payload_path": str(payload_path)}

    # Build a static chain of tasks per slice so the Airflow scheduler sees
    # concrete identities (``train_slice_5k``, ``benchmark_5k``, ...) rather
    # than dynamic mapping that this DAG factory does not implement.
    previous = validate(ingest())
    configuration_for_slices = _dataset_sizing_from_config(config_path)
    verified_outputs: list = []
    for sample_size in configuration_for_slices:
        trained = train_slice.override(task_id=f"train_slice_{sample_size}")(previous, sample_size)
        optimized = export_onnx_for_slice.override(task_id=f"export_onnx_{sample_size}")(trained)
        benchmarked = benchmark_for_slice.override(task_id=f"benchmark_{sample_size}")(optimized)
        verified = verify.override(task_id=f"verify_{sample_size}")(benchmarked, sample_size)
        verified_outputs.append(verified)

    @task(execution_timeout=timedelta(minutes=5))
    def compare_slices(verified_payloads: Iterable[dict]) -> dict:
        rows: list[dict] = []
        for payload in verified_payloads:
            payload_path = Path(payload["payload_path"])
            if not payload_path.is_file():
                continue
            slice_payload = json.loads(payload_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "sample_size": slice_payload.get("sample_size"),
                    "model_version": slice_payload.get("model_version"),
                    "sklearn": slice_payload.get("sklearn"),
                    "onnx": slice_payload.get("onnx"),
                }
            )
        ordered = sorted(rows, key=lambda row: row["sample_size"] or 0)
        out_path = (
            Path(os.environ.get("TRIAGE_REPORTS_DIR", "/opt/triage-ml/reports"))
            / "benchmarks"
            / "dataset_sizing.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(out_path, ordered)
        return {"dataset_sizing_path": str(out_path), "rows": ordered}

    compare_slices(verified_outputs)


triage_ml_retraining_optimization()
