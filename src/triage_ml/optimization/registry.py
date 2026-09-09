"""Variant registry for the optimization pipeline (Fase 2, Etapa 5).

Maps ``MODEL_VARIANT`` env values to the loader that materializes a runtime
"predictor" object from a stored artifact directory. A "predictor" is any
object exposing ``predict``, optional ``predict_proba`` and ``classes_``.

The registry keeps the API official decoupled from the optimization extras:
``resolve_variant_loader("sklearn")`` works without ``onnxruntime`` installed,
and the ONNX loader is only invoked when the operator sets
``TRIAGE_ML_MODEL_VARIANT=onnx`` and the metadata advertises an ONNX variant.
"""

from __future__ import annotations

import hmac
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from triage_ml.models.artifact import (
    ArtifactCompatibilityError,
    ensure_no_symlink_ancestor,
    file_sha256,
    validate_artifact_bundle,
)
from triage_ml.optimization.onnx_adapter import OnnxModelAdapter, normalize_classifier_kind

VariantName = Literal["sklearn", "onnx"]
AVAILABLE_VARIANTS: tuple[VariantName, ...] = ("sklearn", "onnx")


def resolve_variant_loader(variant: str) -> Callable[[Path], Any]:
    """Return a function that loads a predictor from ``models/<version>/``.

    The returned loader expects a path to ``model.joblib`` (sklearn) or
    ``model.onnx`` (onnx). Loading is lazy; the registry does not import
    ONNX at registration time.
    """

    if variant == "sklearn":
        return _load_sklearn
    if variant == "onnx":
        return _load_onnx
    raise ValueError(
        f"unknown TRIAGE_ML_MODEL_VARIANT {variant!r}; valid: {list(AVAILABLE_VARIANTS)}"
    )


def _load_sklearn(joblib_path: Path) -> Any:
    import joblib

    return joblib.load(joblib_path)


def _load_onnx(onnx_path: Path | str) -> OnnxModelAdapter:
    """Load an ``OnnxModelAdapter`` from ``model.onnx`` + ``metadata.json``.

    Defensive against malformed manifests: classes is coerced via the
    public ``validate_artifact_bundle`` (when available) or, failing that,
    via a strict ``(metadata["classes"] or [])`` cast that raises with a
    helpful error if the manifest has zero or non-int labels.
    """

    onnx_path = Path(onnx_path)
    ensure_no_symlink_ancestor(onnx_path)
    if onnx_path.is_symlink() or not onnx_path.is_file():
        raise ArtifactCompatibilityError("model.onnx must be a regular file")
    metadata_path = onnx_path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    classes_raw = metadata.get("classes") or []
    if not isinstance(classes_raw, list) or not classes_raw:
        raise ValueError(
            f"metadata.json at {metadata_path} does not declare a non-empty 'classes' "
            f"list; got {classes_raw!r}"
        )
    if any(not isinstance(label, int) or isinstance(label, bool) for label in classes_raw):
        raise ValueError("metadata.classes must contain only integer labels")
    classes: tuple[int, ...] = tuple(classes_raw)
    if len(classes) != len(set(classes)):
        raise ValueError("metadata.classes must contain unique labels")

    validated = validate_artifact_bundle(onnx_path.with_name("model.joblib"))
    validate_variant_metadata(validated, variant="onnx")
    optimization = validated.get("optimization")
    if not isinstance(optimization, dict):
        raise ArtifactCompatibilityError("metadata.optimization is required for the ONNX variant")
    expected_checksum = optimization.get("onnx_checksum_sha256")
    source_checksum = optimization.get("source_joblib_checksum_sha256")
    if not isinstance(expected_checksum, str) or not hmac.compare_digest(
        file_sha256(onnx_path), expected_checksum
    ):
        raise ArtifactCompatibilityError("model.onnx checksum does not match metadata")
    if not isinstance(source_checksum, str) or not hmac.compare_digest(
        source_checksum, validated["checksum_sha256"]
    ):
        raise ArtifactCompatibilityError("model.onnx was not exported from the active model.joblib")

    preprocessing_classifier = validated.get("preprocessing", {}).get("classifier")
    classifier_kind = normalize_classifier_kind(preprocessing_classifier)

    return OnnxModelAdapter(
        onnx_path=onnx_path,
        classes=classes,
        classifier_kind=classifier_kind,
    )


def validate_variant_metadata(metadata: dict[str, Any], *, variant: str) -> None:
    """Raise ``ValueError`` when ``metadata`` does not advertise ``variant``."""

    available = metadata.get("available_variants")
    if not isinstance(available, list) or variant not in available:
        raise ValueError(
            f"metadata does not advertise variant {variant!r}; available: {available!r}"
        )


def read_runtime_variant(default: str = "sklearn") -> VariantName:
    """Resolve the runtime variant from ``TRIAGE_ML_MODEL_VARIANT``."""

    candidate = os.environ.get("TRIAGE_ML_MODEL_VARIANT", default).lower()
    if candidate not in AVAILABLE_VARIANTS:
        raise ValueError(
            f"TRIAGE_ML_MODEL_VARIANT must be one of {AVAILABLE_VARIANTS}; got {candidate!r}"
        )
    return candidate  # type: ignore[return-value]


def resolve_artifact_for_variant(
    model_version_dir: Path,
    variant: str,
    *,
    onnx_filename: str = "model.onnx",
    joblib_filename: str = "model.joblib",
) -> Path:
    """Return the artifact file used for ``variant`` inside ``model_version_dir``."""

    if variant == "sklearn":
        return model_version_dir / joblib_filename
    if variant == "onnx":
        return model_version_dir / onnx_filename
    raise ValueError(f"unknown variant {variant!r}")
