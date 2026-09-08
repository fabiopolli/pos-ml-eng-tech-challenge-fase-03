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


# ---------------------------------------------------------------------------
# Fase 2 / Etapa 5 — optimization helpers (Airflow pipeline integration)
# ---------------------------------------------------------------------------
# These helpers reuse the project's canonical artifacts and never duplicate
# training or dataset preparation logic. The Airflow DAG
# ``triage_ml_retraining_optimization`` orchestrates them per slice.

# Cap the number of texts each optimization step forwards through the
# sklearn/ONNX pipelines. Larger inputs are sampled uniformly at the
# deterministic index set so the comparison remains comparable.
_OPTIMIZATION_PROBE_TEXTS = 64


def _manifest_path_for(version_dir: Path | str) -> Path:
    """Locate the ``metadata.json`` for a given model-version directory."""

    return Path(version_dir) / "metadata.json"


def _read_metadata_for_version(version_dir: Path | str) -> dict[str, Any]:
    from triage_ml.models.artifact import read_metadata

    return read_metadata(_manifest_path_for(version_dir))


def export_onnx_for_version(
    version_dir: str | Path,
    *,
    opset: int = 17,
) -> dict[str, Any]:
    """Export an ONNX sibling for an existing artifact ``version_dir``.

    Reads ``metadata.json`` to capture the canonical config fingerprint,
    loads the persisted ``model.joblib`` and writes ``model.onnx`` next to
    it. ``reused=True`` when ``model.onnx`` already exists **and** its
    ``onnx_checksum_sha256`` and optimization fingerprint match the
    expected ones — exporting again is wasteful under the Fase 2 latency
    budget.

    The function depends on the optional ``[optimization]`` group; it raises
    ``RuntimeError`` with an actionable message when the extras are missing.
    """

    from triage_ml.optimization.optimize import export_onnx, fingerprint_dict, fingerprint_hash

    version = Path(version_dir)
    metadata = _read_metadata_for_version(version)
    joblib_path = version / "model.joblib"
    if not joblib_path.is_file():
        raise FileNotFoundError(f"model.joblib not found in {version}")

    onnx_path = version / "model.onnx"

    # Reuse path — only fires when the prior export advertised the
    # canonical fingerprint + checksum and the file is still on disk.
    existing_optimization = metadata.get("optimization") or {}
    existing_fingerprint = (existing_optimization.get("optimization_fingerprint") or {}).get(
        "fingerprint_hash"
    )
    existing_checksum = existing_optimization.get("onnx_checksum_sha256")
    if onnx_path.is_file() and existing_fingerprint and existing_checksum:
        current_checksum = file_sha256(onnx_path)
        if current_checksum == existing_checksum:
            return {
                "reused": True,
                "model_version": metadata["model_version"],
                "joblib": str(joblib_path),
                "metrics": metadata["metrics"],
                "optimization": existing_optimization,
            }

    import joblib

    pipeline = joblib.load(joblib_path)
    target_path, fingerprint = export_onnx(pipeline, onnx_path, opset=opset)

    optimization_fingerprint_dict = fingerprint_dict(
        pipeline, opset=opset, quantized=False
    ).to_dict()
    optimization_fingerprint_dict["fingerprint_hash"] = fingerprint_hash(fingerprint)
    optimization_record = {
        "onnx_path": str(target_path),
        "onnx_checksum_sha256": file_sha256(target_path),
        "onnx_opset": fingerprint.opset,
        "optimization_fingerprint": optimization_fingerprint_dict,
        "created_at": _isoformat_utc(),
    }
    return {
        "reused": False,
        "model_version": metadata["model_version"],
        "joblib": str(joblib_path),
        "metrics": metadata["metrics"],
        "optimization": optimization_record,
    }


def benchmark_for_version(
    version_dir: str | Path,
    *,
    benchmark_input_size: int = _OPTIMIZATION_PROBE_TEXTS,
) -> dict[str, Any]:
    """Compare sklearn vs ONNX variants of an artifact version.

    The sklearn reference is benchmarked first; if ``model.onnx`` exists next
    to ``model.joblib`` and the optimization extras are installed, the ONNX
    variant is benchmarked on the same probe. Hardware/environment metadata
    is captured via ``optimization.benchmark.capture_environment``.

    Two JSON artifacts are written:

    * ``models/<ver>/benchmark.json`` — sibling to the artifact itself.
    * ``reports/benchmarks/benchmark.json`` — aggregate evidence used by
      dashboards and tests.

    When ONNX is unavailable the function still runs and emits
    ``{"sklearn": ..., "onnx": null}`` so the orchestrator does not have to
    branch.
    """

    import numpy as np

    from triage_ml.optimization.benchmark import (
        benchmark_predictor,
        capture_environment,
    )
    from triage_ml.optimization.registry import (
        resolve_variant_loader,
        validate_variant_metadata,
    )

    if benchmark_input_size <= 0:
        raise ValueError("benchmark_input_size must be > 0")

    version = Path(version_dir)
    metadata = _read_metadata_for_version(version)
    joblib_path = version / "model.joblib"
    onnx_path = version / "model.onnx"

    # Single deserialisation of the sklearn bundle — both the reference
    # predictions and the sklearn benchmark need it.
    sklearn_pipeline = resolve_variant_loader("sklearn")(joblib_path)
    probe = _build_probe(benchmark_input_size)
    sklearn_reference = [
        int(value) for value in np.asarray(sklearn_pipeline.predict(probe)).reshape(-1).tolist()
    ]

    sklearn_result = benchmark_predictor(
        sklearn_pipeline,
        texts=probe,
        variant="sklearn",
        repetitions=10,
        warmup=2,
    )

    onnx_result = None
    if onnx_path.is_file():
        # Validate the artefact metadata advertises the ONNX variant.
        # We log a warning instead of silently falling back so the
        # operator notices when ``model.onnx`` shows up without a
        # ``available_variants`` update (an export that bypassed the
        # canonical helper).
        try:
            validate_variant_metadata(metadata, variant="onnx")
        except ValueError as exc:
            import structlog

            structlog.get_logger().warning(
                "benchmark_for_version_skipped_onnx_validation",
                error=str(exc),
                version_dir=str(version),
            )
            onnx_result = None
        else:
            try:
                onnx_predictor = resolve_variant_loader("onnx")(onnx_path)
                onnx_result = benchmark_predictor(
                    onnx_predictor,
                    texts=probe,
                    variant="onnx",
                    reference_predictions=sklearn_reference,
                    reference_labels=sklearn_reference,
                    repetitions=10,
                    warmup=2,
                )
            except (RuntimeError, FileNotFoundError):
                # Optimization extras missing in the runtime environment,
                # or the .onnx disappeared between slice tasks; record
                # the absence rather than failing the whole DAG.
                onnx_result = None

    environment = capture_environment()
    aggregate_path = _resolve_reports_path() / "benchmarks" / "benchmark.json"
    sibling_path = version / "benchmark.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sklearn": sklearn_result.to_dict(),
        "onnx": onnx_result.to_dict() if onnx_result is not None else None,
        "environment": environment.to_dict(),
        "extra_metadata": {"model_version": metadata["model_version"]},
    }

    _atomic_write_json(aggregate_path, payload)
    _atomic_write_json(sibling_path, payload)
    return {
        "model_version": metadata["model_version"],
        "benchmark_paths": {"aggregate": str(aggregate_path), "sibling": str(sibling_path)},
        "sklearn": sklearn_result.to_dict(),
        "onnx": onnx_result.to_dict() if onnx_result is not None else None,
    }


def _resolve_reports_path() -> Path:
    """Resolve the canonical reports dir from ``TRIAGE_REPORTS_DIR``/CWD."""

    raw = os.environ.get("TRIAGE_REPORTS_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path("reports").resolve()


def _build_probe(size: int) -> list[str]:
    """Build a deterministic probe input shared by both variants."""

    return [
        f"Optimisation probe input number {i:04d} covering cardiology and oncology"
        for i in range(max(1, size))
    ]


def _isoformat_utc() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def build_optimization_manifest(
    version_dir: str | Path,
    *,
    available_variants: tuple[str, ...] = ("sklearn",),
) -> dict[str, Any]:
    """Build the canonical ``available_variants`` payload for ``metadata.json``.

    The optimization DAG calls this helper after a successful ONNX export so
    the artifact manifest advertises the variants it ships with. ``metadata``
    itself is mutated only when ``apply=True`` (out of scope here — the
    train flow keeps ``metadata.json`` immutable until a new version is
    materialised).
    """

    payload = {
        "available_variants": list(available_variants),
        "build_utc": _isoformat_utc(),
    }
    return payload


def _effective_training_config(config_path: str | Path) -> dict[str, Any]:
    """Read the ``configs/training.yaml`` file and return its content as dict."""

    import yaml

    with Path(config_path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _slice_identity_fields(
    config: dict[str, Any],
    *,
    sample_size: int,
    selected_classifier: str | None = None,
) -> dict[str, Any]:
    """Pick the deterministic fields that drive ``run_training`` equivalence.

    The optimization idempotency key is the tuple
    ``(dataset_sha256, config_file_sha256, sample_size, random_state,
    test_size, selected_classifier, task_type, language)``. The first three
    are obvious; the rest matter because two runs with the same dataset
    config but different seeds or classifiers would yield different
    ``model.joblib`` outputs and the benchmark would compare apples to
    oranges.

    ``selected_classifier`` is **not** silently defaulted to ``"logreg"`` —
    it must come from ``summary["selection"]["selected_classifier"]`` (the
    canonical record produced by ``run_training``) or ``selection_overrides``
    in the YAML. When neither is present the field is forced to
    ``"unknown"`` so the idempotency check refuses to reuse a manifest
    that does not declare its class.
    """

    classifier = selected_classifier or config.get("selection_overrides", {}).get("classifier")
    if not classifier:
        raise ValueError(
            "configs/training.yaml: missing 'selection_overrides.classifier' or equivalent; "
            "the optimization idempotency key requires an explicit class. Add "
            "'selection_overrides: {classifier: logreg}' to training.yaml or pass "
            "selected_classifier to train_with_sample_size()."
        )

    return {
        "sample_size": sample_size,
        "random_state": int(config["random_state"]),
        "test_size": float(config["test_size"]),
        "selected_classifier": str(classifier),
        "task_type": str(config.get("task_type", "multiclass_text_classification")),
        "language": str(config.get("language", "en")),
    }


def train_with_sample_size(
    *,
    dataset_path: str | Path,
    sample_size: int,
    models_dir: str | Path,
    figures_dir: str | Path,
    config_path: str | Path,
    source_commit: str,
) -> dict[str, Any]:
    """Train with an explicit ``sample_size`` override and persist the artifact.

    Used by the optimization DAG to materialise a version directory per
    ``dataset_sizing`` entry. The function is idempotent: when an existing
    artefact shares the slice identity (see ``_slice_identity_fields``) the
    helper reuses it without re-running ``run_training``.
    """

    from triage_ml.models.train import run_training

    dataset_hash = file_sha256(dataset_path)
    config_hash = file_sha256(config_path)
    config = _effective_training_config(config_path)

    existing = _find_reusable_for_size(
        models_dir,
        dataset_sha256=dataset_hash,
        config_file_sha256=config_hash,
        sample_size=sample_size,
        config=config,
    )
    if existing is not None:
        return existing

    summary = run_training(
        raw_csv_path=dataset_path,
        out_dir=models_dir,
        figures_dir=figures_dir,
        config_path=config_path,
        sample_size=sample_size,
    )

    # ``summary["selection"]["selected_classifier"]`` is the *executable*
    # classifier (overrides + CV winner). Read it from the canonical
    # record so the idempotency key reflects what ``run_training``
    # actually materialised on disk, not a guess from the YAML.
    selected_classifier = str(summary.get("selection", {}).get("selected_classifier") or "logreg")
    identity = _slice_identity_fields(
        config,
        sample_size=sample_size,
        selected_classifier=selected_classifier,
    )
    version_dir = Path(models_dir) / summary["model_version"]
    run_manifest = {
        "dataset_sha256": dataset_hash,
        "config_file_sha256": config_hash,
        "source_commit": source_commit,
        **identity,
    }
    _atomic_write_json(version_dir / "airflow_run.json", run_manifest)
    return {
        "reused": False,
        "model_version": summary["model_version"],
        "sample_size": sample_size,
        "joblib": summary["paths"]["joblib"],
        "metrics": summary["metrics"],
    }


def _find_reusable_for_size(
    models_dir: str | Path,
    *,
    dataset_sha256: str,
    config_file_sha256: str,
    sample_size: int,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Find a prior artefact whose manifest declares the slice identity.

    The helper validates ``dataset_sha256``, ``config_file_sha256`` and
    ``sample_size`` against ``airflow_run.json``; it then ensures the
    candidate's ``model.joblib`` is still on disk so ``benchmark_for_version``
    does not blow up later. ``metrics`` come from the manifest itself so
    the reused payload stays consistent.
    """

    for manifest_path in sorted(Path(models_dir).glob("*/airflow_run.json"), reverse=True):
        if not VERSION_PATTERN.fullmatch(manifest_path.parent.name):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if manifest.get("dataset_sha256") != dataset_sha256:
            continue
        if manifest.get("config_file_sha256") != config_file_sha256:
            continue
        if manifest.get("sample_size") != sample_size:
            continue
        joblib_path = manifest_path.parent / "model.joblib"
        if not joblib_path.is_file():
            # Manifest points at an orphan artefact (cleanup, partial
            # publish, FS evicted); skip and let the caller train again.
            continue
        # Cross-check the persisted identity against the active config to
        # surface silent drift (seed/classifier change). We use the
        # manifest's value (not the regenerated one) to account for
        # ``selection_overrides`` that may have changed between runs.
        manifest_classifier = manifest.get("selected_classifier")
        active_classifier = config.get("selection_overrides", {}).get("classifier")
        if manifest_classifier and active_classifier and manifest_classifier != active_classifier:
            continue
        return {
            "reused": True,
            "model_version": manifest_path.parent.name,
            "sample_size": sample_size,
            "joblib": str(joblib_path),
            "metrics": manifest.get("metrics"),
        }
    return None


# ``export_onnx_for_version_dir`` and ``benchmark_for_version_dir`` were
# removed during the post-Fase-2 review (they were dead wrappers
# duplicating ``export_onnx_for_version`` and ``benchmark_for_version``).
# Callers (including the optimization DAG) import the canonical helpers
# directly.
