"""Artifact layout, manifest validation, and safe local loading."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
VERSION_PATTERN = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{12}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FINGERPRINT_KEYS = {
    "raw_csv_sha256",
    "prepared_dataset_sha256",
    "train_split_sha256",
    "test_split_sha256",
    "config_sha256",
}
REQUIRED_METADATA_KEYS = (
    "schema_version",
    "model_version",
    "model_name",
    "task_type",
    "language",
    "classes",
    "label_mapping",
    "random_state",
    "n_train",
    "n_test",
    "metrics",
    "preprocessing",
    "selection",
    "dependency_versions",
    "git_commit",
    "git_dirty",
    "fingerprints",
    "checksum_sha256",
    "created_at",
)


@dataclass(frozen=True)
class ArtifactPaths:
    """Filesystem layout for a single immutable model version."""

    version_dir: Path
    joblib: Path
    classes: Path
    metadata: Path

    @classmethod
    def for_version(cls, out_dir: str | Path, version: str) -> ArtifactPaths:
        version_dir = Path(out_dir) / version
        return cls(
            version_dir=version_dir,
            joblib=version_dir / "model.joblib",
            classes=version_dir / "classes.json",
            metadata=version_dir / "metadata.json",
        )

    def ensure(self) -> None:
        """Create the version directory and refuse to overwrite an artifact."""

        self.version_dir.mkdir(parents=True, exist_ok=False)


class ArtifactIntegrityError(RuntimeError):
    """Raised when an artifact checksum does not match its manifest."""


class ArtifactCompatibilityError(RuntimeError):
    """Raised when model contents disagree with the validated manifest."""


def file_sha256(path: Path) -> str:
    """Return the lowercase hexadecimal SHA-256 of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_classes(path: Path, classes: Iterable[Any]) -> list[Any]:
    class_list = [_coerce(value) for value in classes]
    path.write_text(json.dumps(class_list, indent=2, ensure_ascii=False), encoding="utf-8")
    return class_list


def read_classes(path: Path) -> list[Any]:
    classes = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(classes, list):
        raise ArtifactCompatibilityError("classes.json must contain a list")
    return classes


def build_metadata(
    *,
    model_version: str,
    model_name: str,
    task_type: str,
    language: str,
    classes: list[Any],
    label_mapping: dict[str, str],
    random_state: int,
    n_train: int,
    n_test: int,
    metrics: dict[str, Any],
    preprocessing: dict[str, Any],
    selection: dict[str, Any],
    dependency_versions: dict[str, str],
    git_commit: str,
    git_dirty: bool,
    fingerprints: dict[str, str],
    joblib_path: Path,
) -> dict[str, Any]:
    """Build the canonical metadata manifest and stamp the model checksum."""

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model_version": model_version,
        "model_name": model_name,
        "task_type": task_type,
        "language": language,
        "classes": [_coerce(value) for value in classes],
        "label_mapping": {str(key): str(value) for key, value in label_mapping.items()},
        "random_state": int(random_state),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "metrics": _coerce(metrics),
        "preprocessing": _coerce(preprocessing),
        "selection": _coerce(selection),
        "dependency_versions": dict(dependency_versions),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "fingerprints": dict(fingerprints),
        "checksum_sha256": file_sha256(joblib_path),
        "created_at": datetime.now(UTC).isoformat(),
    }
    validate_metadata(metadata)
    return metadata


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    validate_metadata(metadata)
    path.write_text(json.dumps(_coerce(metadata), indent=2, ensure_ascii=False), encoding="utf-8")


def _coerce(obj: Any) -> Any:
    """Recursively convert numpy and pandas scalars to native Python types."""

    if isinstance(obj, dict):
        return {str(key): _coerce(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce(value) for value in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except (ValueError, TypeError):
            return obj
    return obj


def read_metadata(path: Path) -> dict[str, Any]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain an object")
    return metadata


def validate_metadata(metadata: dict[str, Any]) -> None:
    """Validate the artifact manifest before any model deserialization."""

    missing = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"metadata.json is missing required keys: {missing}")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported metadata schema_version: {metadata['schema_version']!r}")
    if not isinstance(metadata["model_version"], str) or not VERSION_PATTERN.fullmatch(
        metadata["model_version"]
    ):
        raise ValueError("metadata.model_version has an invalid format")
    for key in ("model_name", "task_type", "language"):
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise ValueError(f"metadata.{key} must be a non-empty string")
    if metadata["task_type"] != "multiclass_text_classification" or metadata["language"] != "en":
        raise ValueError("metadata task_type or language is unsupported")

    classes = metadata["classes"]
    if (
        not isinstance(classes, list)
        or not classes
        or any(not isinstance(value, int) or isinstance(value, bool) for value in classes)
        or len(classes) != len(set(classes))
    ):
        raise ValueError("metadata.classes must contain unique integer labels")
    label_mapping = metadata["label_mapping"]
    if not isinstance(label_mapping, dict) or set(label_mapping) != {
        str(value) for value in classes
    }:
        raise ValueError("metadata.label_mapping must cover exactly metadata.classes")
    if any(not isinstance(value, str) or not value.strip() for value in label_mapping.values()):
        raise ValueError("metadata.label_mapping names must be non-empty strings")

    for key in ("random_state", "n_train", "n_test"):
        if not isinstance(metadata[key], int) or isinstance(metadata[key], bool):
            raise ValueError(f"metadata.{key} must be an integer")
    if metadata["n_train"] <= 0 or metadata["n_test"] <= 0:
        raise ValueError("metadata split sizes must be positive")
    for key in ("metrics", "preprocessing", "selection", "dependency_versions"):
        if not isinstance(metadata[key], dict) or not metadata[key]:
            raise ValueError(f"metadata.{key} must be a non-empty object")
    metrics = metadata["metrics"]
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
        if metric not in metrics:
            raise ValueError(f"metadata.metrics is missing {metric}")
        value = metrics[metric]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"metadata.metrics.{metric} must be between zero and one")
    per_class = metrics.get("per_class")
    if not isinstance(per_class, dict) or set(per_class) != {str(value) for value in classes}:
        raise ValueError("metadata.metrics.per_class must cover exactly metadata.classes")
    for class_metrics in per_class.values():
        if not isinstance(class_metrics, dict):
            raise ValueError("metadata per-class metrics must be objects")
        for metric in ("precision", "recall", "f1"):
            value = class_metrics.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= 1
            ):
                raise ValueError(f"metadata per-class {metric} must be between zero and one")
        support = class_metrics.get("support")
        if isinstance(support, bool) or not isinstance(support, int) or support < 0:
            raise ValueError("metadata per-class support must be a non-negative integer")

    selection = metadata["selection"]
    candidates = selection.get("candidates")
    folds = selection.get("folds")
    if (
        not isinstance(candidates, dict)
        or set(candidates) != {"logreg", "linear_svc"}
        or selection.get("selected_classifier") not in candidates
        or selection.get("test_set_used_for_selection") is not False
        or selection.get("metric") != "macro_f1"
        or isinstance(folds, bool)
        or not isinstance(folds, int)
        or folds < 2
    ):
        raise ValueError("metadata.selection must describe training-only candidate selection")
    for candidate in candidates.values():
        fold_scores = candidate.get("fold_macro_f1") if isinstance(candidate, dict) else None
        if (
            not isinstance(fold_scores, list)
            or len(fold_scores) != folds
            or any(
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not 0 <= score <= 1
                for score in fold_scores
            )
        ):
            raise ValueError("metadata candidate fold scores are invalid")
        for key in ("mean_macro_f1", "std_macro_f1"):
            value = candidate.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"metadata candidate {key} is invalid")

    preprocessing = metadata["preprocessing"]
    if (
        preprocessing.get("classifier") != selection["selected_classifier"]
        or not isinstance(preprocessing.get("tfidf"), dict)
        or not isinstance(preprocessing.get("classifier_params"), dict)
    ):
        raise ValueError("metadata preprocessing disagrees with model selection")
    dependencies = metadata["dependency_versions"]
    required_dependencies = {"python", "numpy", "scipy", "scikit_learn", "joblib"}
    if not required_dependencies.issubset(dependencies) or any(
        not isinstance(dependencies[name], str) or not dependencies[name]
        for name in required_dependencies
    ):
        raise ValueError("metadata dependency_versions is incomplete")

    fingerprints = metadata["fingerprints"]
    if not isinstance(fingerprints, dict) or not FINGERPRINT_KEYS.issubset(fingerprints):
        raise ValueError("metadata.fingerprints is incomplete")
    if any(
        not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value)
        for value in fingerprints.values()
    ):
        raise ValueError("metadata fingerprints must be lowercase SHA-256 digests")
    checksum = metadata["checksum_sha256"]
    if not isinstance(checksum, str) or not SHA256_PATTERN.fullmatch(checksum):
        raise ValueError("metadata.checksum_sha256 must be a lowercase SHA-256 digest")
    git_commit = metadata["git_commit"]
    if git_commit != "unknown" and (
        not isinstance(git_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", git_commit)
    ):
        raise ValueError("metadata.git_commit must be a full Git SHA or 'unknown'")
    if not isinstance(metadata["git_dirty"], bool):
        raise ValueError("metadata.git_dirty must be a boolean")
    try:
        datetime.fromisoformat(metadata["created_at"])
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata.created_at must be an ISO-8601 timestamp") from exc


def verify_artifact_integrity(*, joblib_path: Path, metadata: dict[str, Any]) -> None:
    """Check corruption or an accidental artifact/manifest mismatch."""

    expected = metadata.get("checksum_sha256")
    if not isinstance(expected, str) or not SHA256_PATTERN.fullmatch(expected):
        raise ArtifactIntegrityError("metadata.json has an invalid checksum_sha256")
    actual = file_sha256(joblib_path)
    if not hmac.compare_digest(actual, expected):
        raise ArtifactIntegrityError(
            f"checksum mismatch for {joblib_path}: expected {expected}, got {actual}"
        )


def load_artifact(joblib_path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Validate a trusted local artifact before deserializing its joblib payload."""

    import joblib

    joblib_path = Path(joblib_path)
    metadata_path = joblib_path.with_name("metadata.json")
    classes_path = joblib_path.with_name("classes.json")
    if not joblib_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"model.joblib and metadata.json are required under {joblib_path.parent}"
        )

    metadata = read_metadata(metadata_path)
    validate_metadata(metadata)
    if joblib_path.parent.name != metadata["model_version"]:
        raise ArtifactCompatibilityError("artifact directory does not match metadata.model_version")
    expected = metadata["checksum_sha256"]
    with joblib_path.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
        actual = digest.hexdigest()
        if not hmac.compare_digest(actual, expected):
            raise ArtifactIntegrityError(
                f"checksum mismatch for {joblib_path}: expected {expected}, got {actual}"
            )
        handle.seek(0)
        pipeline = joblib.load(handle)

    model_classes = [_coerce(value) for value in getattr(pipeline, "classes_", [])]
    if model_classes != metadata["classes"]:
        raise ArtifactCompatibilityError("model classes do not match metadata.classes")
    if classes_path.exists() and read_classes(classes_path) != metadata["classes"]:
        raise ArtifactCompatibilityError("classes.json does not match metadata.classes")
    return pipeline, metadata
