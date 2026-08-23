"""Artifact layout, manifest validation, and safe local loading."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import platform
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
    path.write_text(
        json.dumps(class_list, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return class_list


def read_classes(path: Path) -> list[Any]:
    classes = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(classes, list)
        or not classes
        or any(not isinstance(value, int) or isinstance(value, bool) for value in classes)
        or len(classes) != len(set(classes))
    ):
        raise ArtifactCompatibilityError("classes.json must contain unique integer labels")
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
    path.write_text(
        json.dumps(_coerce(metadata), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


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


def validate_model_version(version: object) -> str:
    """Validate both the lexical schema and the UTC timestamp of a model version."""

    if not isinstance(version, str) or not VERSION_PATTERN.fullmatch(version):
        raise ValueError("metadata.model_version has an invalid format")
    try:
        datetime.strptime(version.split("-", 1)[0], "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise ValueError("metadata.model_version has an invalid timestamp") from exc
    return version


def validate_metadata(metadata: dict[str, Any]) -> None:
    """Validate the artifact manifest before any model deserialization."""

    missing = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"metadata.json is missing required keys: {missing}")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported metadata schema_version: {metadata['schema_version']!r}")
    validate_model_version(metadata["model_version"])
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
    if sum(class_metrics["support"] for class_metrics in per_class.values()) != metadata["n_test"]:
        raise ValueError("metadata per-class supports must sum to metadata.n_test")

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
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (key == "mean_macro_f1" and value > 1)
            ):
                raise ValueError(f"metadata candidate {key} is invalid")
        mean = math.fsum(fold_scores) / len(fold_scores)
        std = math.sqrt(math.fsum((score - mean) ** 2 for score in fold_scores) / len(fold_scores))
        if not math.isclose(candidate["mean_macro_f1"], mean, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("metadata candidate mean_macro_f1 disagrees with fold scores")
        if not math.isclose(candidate["std_macro_f1"], std, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("metadata candidate std_macro_f1 disagrees with fold scores")

    best_classifier = max(candidates, key=lambda name: candidates[name]["mean_macro_f1"])
    if selection.get("best_classifier") != best_classifier:
        raise ValueError("metadata.selection.best_classifier is inconsistent")
    policy = selection.get("selection_policy")
    selected = selection["selected_classifier"]
    if policy not in {"highest_mean_macro_f1", "explicit_override"} or (
        policy == "highest_mean_macro_f1" and selected != best_classifier
    ):
        raise ValueError("metadata.selection.selection_policy is inconsistent")

    preprocessing = metadata["preprocessing"]
    if (
        preprocessing.get("vectorizer") != "tfidf"
        or preprocessing.get("classifier") != selection["selected_classifier"]
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
        created_at = datetime.fromisoformat(metadata["created_at"])
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata.created_at must be an ISO-8601 timestamp") from exc
    if created_at.utcoffset() != UTC.utcoffset(created_at):
        raise ValueError("metadata.created_at must be an explicit UTC timestamp")


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


def validate_artifact_bundle(joblib_path: str | Path) -> dict[str, Any]:
    """Validate a trusted local artifact without deserializing its joblib payload."""

    import joblib
    import numpy
    import scipy
    import sklearn

    joblib_path = Path(joblib_path)
    metadata_path = joblib_path.with_name("metadata.json")
    classes_path = joblib_path.with_name("classes.json")
    if (
        joblib_path.parent.is_symlink()
        or joblib_path.is_symlink()
        or metadata_path.is_symlink()
        or classes_path.is_symlink()
    ):
        raise ArtifactCompatibilityError("artifact symlinks are not allowed")
    if not joblib_path.is_file() or not metadata_path.is_file() or not classes_path.is_file():
        raise FileNotFoundError(
            f"model.joblib, metadata.json and classes.json are required under {joblib_path.parent}"
        )

    metadata = read_metadata(metadata_path)
    validate_metadata(metadata)
    if joblib_path.parent.name != metadata["model_version"]:
        raise ArtifactCompatibilityError("artifact directory does not match metadata.model_version")
    runtime_versions = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    mismatches: dict[str, tuple[str, str]] = {}
    for name, runtime_version in runtime_versions.items():
        trained_version = metadata["dependency_versions"][name]
        compatible = (
            trained_version == runtime_version
            if name == "scikit_learn"
            else trained_version.split(".")[:2] == runtime_version.split(".")[:2]
        )
        if not compatible:
            mismatches[name] = (trained_version, runtime_version)
    if mismatches:
        raise ArtifactCompatibilityError(
            f"artifact dependency versions are incompatible with the runtime: {mismatches}"
        )
    verify_artifact_integrity(joblib_path=joblib_path, metadata=metadata)
    if read_classes(classes_path) != metadata["classes"]:
        raise ArtifactCompatibilityError("classes.json does not match metadata.classes")
    return metadata


def load_artifact(joblib_path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Validate a trusted local artifact before deserializing its joblib payload."""

    import joblib

    joblib_path = Path(joblib_path)
    metadata = validate_artifact_bundle(joblib_path)
    payload = joblib_path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual, metadata["checksum_sha256"]):
        raise ArtifactIntegrityError("model.joblib changed after bundle validation")
    pipeline = joblib.load(io.BytesIO(payload))

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.utils.validation import check_is_fitted

    if not isinstance(pipeline, Pipeline) or list(pipeline.named_steps) != ["tfidf", "clf"]:
        raise ArtifactCompatibilityError("model must be a tfidf/clf sklearn Pipeline")
    tfidf = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    expected_classifier = metadata["preprocessing"]["classifier"]
    classifier_types = {"logreg": LogisticRegression, "linear_svc": LinearSVC}
    if not isinstance(tfidf, TfidfVectorizer) or not isinstance(
        classifier, classifier_types[expected_classifier]
    ):
        raise ArtifactCompatibilityError("model steps disagree with metadata.preprocessing")
    try:
        check_is_fitted(tfidf)
        check_is_fitted(classifier)
    except Exception as exc:
        raise ArtifactCompatibilityError("model pipeline contains unfitted steps") from exc
    classifier_features = getattr(classifier, "n_features_in_", None)
    if classifier_features != len(tfidf.vocabulary_):
        raise ArtifactCompatibilityError("model pipeline steps have incompatible dimensions")
    declared_params = (
        ("tfidf", metadata["preprocessing"]["tfidf"], tfidf.get_params(deep=False)),
        (
            "classifier",
            metadata["preprocessing"]["classifier_params"],
            classifier.get_params(deep=False),
        ),
    )
    for component, expected_params, actual_params in declared_params:
        for name, expected_value in expected_params.items():
            if name not in actual_params or _coerce(actual_params[name]) != _coerce(expected_value):
                raise ArtifactCompatibilityError(
                    f"model {component} parameter {name!r} disagrees with metadata"
                )

    model_classes = [_coerce(value) for value in getattr(pipeline, "classes_", [])]
    if model_classes != metadata["classes"]:
        raise ArtifactCompatibilityError("model classes do not match metadata.classes")
    return pipeline, metadata
