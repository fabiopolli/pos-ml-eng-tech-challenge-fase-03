"""Read and write the serialized artifact metadata.

The artifact follows the project contract defined in
``.agents/contracts/README.md``:

- ``model.joblib``: the scikit-learn pipeline fitted on the canonical split.
- ``classes.json``: ordered list of class labels exactly matching
  ``pipeline.classes_``.
- ``metadata.json``: provenance, versions, preprocessing, metrics and
  deterministic fingerprint. ``metadata.classes`` must equal
  ``pipeline.classes_`` and the recorded SHA-256 of the joblib file must
  match the actual artifact, otherwise loading is rejected.

The functions here operate on plain dictionaries so they can be used
both at training time (writer) and at inference time (reader).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REQUIRED_METADATA_KEYS = (
    "model_version",
    "model_name",
    "classes",
    "sklearn_version",
    "numpy_version",
    "python_version",
    "random_state",
    "n_train",
    "n_test",
    "metrics",
    "preprocessing",
    "checksum_sha256",
    "created_at",
)


@dataclass(frozen=True)
class ArtifactPaths:
    """Filesystem layout for a single model version."""

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
        self.version_dir.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path) -> str:
    """Return the lowercase hex SHA-256 of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_classes(path: Path, classes: Iterable[Any]) -> list[Any]:
    """Persist ``classes`` as a JSON list and return the materialized list."""

    class_list = [_coerce(c) for c in classes]
    path.write_text(json.dumps(class_list, indent=2, ensure_ascii=False), encoding="utf-8")
    return class_list


def read_classes(path: Path) -> list[Any]:
    """Load the persisted classes list."""

    return json.loads(path.read_text(encoding="utf-8"))


def build_metadata(
    *,
    model_version: str,
    model_name: str,
    classes: list[Any],
    random_state: int,
    n_train: int,
    n_test: int,
    metrics: dict[str, Any],
    preprocessing: dict[str, Any],
    joblib_path: Path,
    sklearn_version: str,
    numpy_version: str,
    python_version: str,
) -> dict[str, Any]:
    """Build the metadata payload and stamp the artifact checksum."""

    return {
        "model_version": model_version,
        "model_name": model_name,
        "classes": list(classes),
        "sklearn_version": sklearn_version,
        "numpy_version": numpy_version,
        "python_version": python_version,
        "random_state": random_state,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "metrics": dict(metrics),
        "preprocessing": dict(preprocessing),
        "checksum_sha256": file_sha256(joblib_path),
        "created_at": datetime.now(UTC).isoformat(),
    }


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(_coerce(metadata), indent=2, ensure_ascii=False), encoding="utf-8")


def _coerce(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars to native Python types."""

    if isinstance(obj, dict):
        return {k: _coerce(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except (ValueError, TypeError):
            return obj
    return obj


def read_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_metadata(metadata: dict[str, Any]) -> None:
    """Raise ``ValueError`` if required keys are missing."""

    missing = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"metadata.json is missing required keys: {missing}")
